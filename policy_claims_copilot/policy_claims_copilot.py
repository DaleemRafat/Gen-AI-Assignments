# Assignment 5 - Policy & Claims Copilot using RAG
#
# Use Case: Build a chatbot that answers insurance policy questions and
# does claim pre-checks for SBI General Insurance customers.
#
# What is RAG?
# RAG = Retrieval Augmented Generation
# Instead of just asking an LLM questions (which might hallucinate answers),
# we first find the relevant parts of the actual policy PDF,
# then give those parts to the LLM along with the question.
# This way the LLM has to answer from the real document, not from memory.
#
# How it works step by step:
# 1. Download the SBI General Insurance policy PDF
# 2. Split the PDF text into overlapping chunks
# 3. Convert each chunk into an embedding (a vector of numbers that captures meaning)
# 4. When user asks a question, convert it to embedding too
# 5. Find the most similar chunks using cosine similarity
# 6. Send those chunks + question to Gemini API
# 7. Get answer back with page citations
#
# LLM used: Google Gemini 2.5 Flash (free tier, via OpenAI-compatible API)
# Embeddings: sentence-transformers (all-MiniLM-L6-v2)
#
# Requirements: pip install pypdf sentence-transformers requests

import os
import sys
import json
import pickle
import re
import time
import getpass
import textwrap
import warnings
from pathlib import Path

import numpy as np
import requests

warnings.filterwarnings("ignore")

# try pypdf for reading the PDF
try:
    from pypdf import PdfReader
    _PDF_BACKEND = "pypdf"
except ImportError:
    try:
        import fitz as _pymupdf_fitz
        _PDF_BACKEND = "pymupdf"
    except ImportError:
        print("\n[ERROR] No PDF library. Run: pip install pypdf")
        sys.exit(1)

# try sentence-transformers for semantic embeddings, fall back to TF-IDF
try:
    from sentence_transformers import SentenceTransformer
    _EMBED_BACKEND = "sbert"
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _tfidf_cosine
    _EMBED_BACKEND = "tfidf"
    print("[WARN] sentence-transformers not installed, using TF-IDF instead.")
    print("       Install it with: pip install sentence-transformers\n")


# ------------------------------------------------------------------
# Configuration
# All settings in one place, easy to change
# ------------------------------------------------------------------

# Google Gemini API (using OpenAI-compatible endpoint)
DEPLOYMENT  = "gemini-2.5-flash"
API_URL     = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
MAX_TOKENS  = 1200
TEMPERATURE = 0.10   # keep it low so answers are factual, not creative

# Source PDF
PDF_URL   = "https://content.sbigeneral.in/uploads/a370272f732749999e7c19e82e38ad7c.pdf"
PDF_LOCAL = "sbi_general_policy.pdf"

# Embedding/chunking settings
EMBED_MODEL   = "all-MiniLM-L6-v2"   # ~90 MB download on first run
CHUNK_WORDS   = 350    # how many words per chunk
OVERLAP_WORDS = 60     # how many words overlap between consecutive chunks
TOP_K         = 5      # how many chunks to retrieve for each query

# Cache files (so we don't re-process the PDF every time)
CHUNK_CACHE = "policy_chunks.pkl"
EMBED_CACHE = "policy_embeddings.pkl"

DIVIDER = "-" * 70


# ------------------------------------------------------------------
# System Prompt
# This tells Gemini what it should be and how it should behave
# I spent a while getting these rules right to prevent it from making things up
# ------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Policy & Claims Copilot for SBI General Insurance.
Your role: give accurate, grounded answers to customers, insurance agents,
and claims teams using ONLY the policy text provided to you.

You can help with:
  - What is covered and what is NOT covered
  - Coverage limits, sub-limits, and deductibles
  - Waiting periods (initial, specific illness, pre-existing disease)
  - Claim submission steps and timelines
  - Documents required for each claim type
  - Pre-checking whether a described claim scenario is eligible for coverage

STRICT RULES - follow these on every single response:
1. Use ONLY the [POLICY CONTEXT] excerpts provided. Do not use general
   insurance knowledge or make assumptions.
2. Cite every fact: include page number in parentheses like (Page 12).
3. If the answer is NOT in the excerpts, say exactly:
   "This information was not found in the retrieved policy sections.
    Please refer to the full policy document or contact SBI General support."
4. For Claim Pre-Check requests, start with ONE verdict:
   [ELIGIBLE] / [POSSIBLY ELIGIBLE] / [NOT ELIGIBLE] / [NEED MORE INFORMATION]
   Then explain with policy citations.
5. Use bullet points for lists (conditions, exclusions, documents, steps).
6. Never guess or extrapolate beyond what the text says.
7. Keep answers clear and avoid unnecessary jargon.
"""


# ------------------------------------------------------------------
# PDF Download
# Downloads the policy PDF and caches it locally
# Handles SSL errors that some insurance company websites have
# ------------------------------------------------------------------

def download_pdf(url, local_path):
    if Path(local_path).exists():
        size_kb = Path(local_path).stat().st_size // 1024
        print(f"[OK] Using cached PDF: {local_path} ({size_kb} KB)")
        return True

    print(f"[>>] Downloading policy PDF from:")
    print(f"     {url}")

    headers = {"User-Agent": "Mozilla/5.0 (RAG-Copilot/1.0)"}

    for verify in (True, False):
        if not verify:
            print("[WARN] Retrying without SSL verification ...")
            import urllib3
            urllib3.disable_warnings()
        try:
            resp = requests.get(url, timeout=90, verify=verify, headers=headers, stream=True)
            resp.raise_for_status()
            data = resp.content
            Path(local_path).write_bytes(data)
            print(f"[OK] PDF saved: {local_path} ({len(data) // 1024} KB)")
            return True
        except requests.exceptions.SSLError:
            continue
        except Exception as exc:
            print(f"[ERROR] Download failed: {exc}")
            break

    print("\nManual download instructions:")
    print(f"  1. Open this URL in your browser: {url}")
    print(f"  2. Save the file as '{local_path}' in the current folder")
    print(f"  3. Run this script again\n")
    return False


# ------------------------------------------------------------------
# Text Extraction from PDF pages
# ------------------------------------------------------------------

def _clean(raw):
    """clean up extracted text - remove lone page numbers, extra spaces"""
    lines = [l.strip() for l in raw.split("\n")
             if l.strip() and not re.fullmatch(r"\d{1,3}", l.strip())]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def extract_pages(pdf_path):
    """
    Extract text from each page of the PDF.
    Returns list of dicts: [{"page": 1, "text": "..."}, ...]
    """
    pages = []

    if _PDF_BACKEND == "pypdf":
        reader = PdfReader(pdf_path)
        total  = len(reader.pages)
        print(f"[>>] Reading {total} pages with pypdf ...")
        for i, pg in enumerate(reader.pages):
            text = _clean(pg.extract_text() or "")
            if text:
                pages.append({"page": i + 1, "text": text})
    else:
        doc   = _pymupdf_fitz.open(pdf_path)
        total = doc.page_count
        print(f"[>>] Reading {total} pages with PyMuPDF ...")
        for i in range(total):
            text = _clean(doc[i].get_text())
            if text:
                pages.append({"page": i + 1, "text": text})
        doc.close()

    print(f"[OK] Got text from {len(pages)} of {total} pages.")
    if len(pages) < total * 0.3:
        print("[WARN] Very few pages have text - PDF might be scanned images (needs OCR)")
    return pages


# ------------------------------------------------------------------
# Chunking - split pages into overlapping fixed-size chunks
# Why overlapping? Policy clauses can span paragraph boundaries.
# If chunks didn't overlap, a sentence right at the boundary might
# get split and neither chunk would match a query about it.
# ------------------------------------------------------------------

def chunk_pages(pages, chunk_words, overlap_words):
    chunks   = []
    chunk_id = 0

    for page_data in pages:
        words    = page_data["text"].split()
        page_num = page_data["page"]

        if len(words) <= chunk_words:
            # whole page fits in one chunk
            chunks.append({
                "chunk_id":   chunk_id,
                "page":       page_num,
                "text":       " ".join(words),
                "word_count": len(words),
            })
            chunk_id += 1
        else:
            # sliding window
            start = 0
            while start < len(words):
                end        = min(start + chunk_words, len(words))
                chunk_text = " ".join(words[start:end])
                chunks.append({
                    "chunk_id":   chunk_id,
                    "page":       page_num,
                    "text":       chunk_text,
                    "word_count": end - start,
                })
                chunk_id += 1
                if end == len(words):
                    break
                start += chunk_words - overlap_words

    total_words = sum(c["word_count"] for c in chunks)
    print(f"[OK] {len(chunks)} chunks created (~{chunk_words} words each, {overlap_words} overlap, {total_words:,} words total)")
    return chunks


# ------------------------------------------------------------------
# VectorStore class
# Handles embedding the chunks and searching for relevant ones
# Two backends: sentence-transformers (semantic) or TF-IDF (keyword)
# ------------------------------------------------------------------

class VectorStore:

    def __init__(self):
        self.chunks     = []
        self.embeddings = None
        self._model     = None
        self._tfidf_vec = None

    def build(self, chunks):
        """Encode all chunks into embeddings"""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        if _EMBED_BACKEND == "sbert":
            print(f"[>>] Loading embedding model '{EMBED_MODEL}' ...")
            print("     (Downloads ~90 MB from HuggingFace on first run)")
            self._model = SentenceTransformer(EMBED_MODEL)
            print(f"[>>] Embedding {len(texts)} chunks ...")
            self.embeddings = self._model.encode(
                texts, batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
        else:
            print(f"[>>] Building TF-IDF index for {len(texts)} chunks ...")
            self._tfidf_vec = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2))
            self.embeddings = self._tfidf_vec.fit_transform(texts)

        print(f"[OK] Done. {len(self.chunks)} chunks embedded.")

    def save(self, chunk_path, embed_path):
        """Save to disk so next run can skip rebuilding"""
        with open(chunk_path, "wb") as fh:
            pickle.dump(self.chunks, fh)
        payload = {
            "backend":    _EMBED_BACKEND,
            "embeddings": self.embeddings,
            "tfidf_vec":  self._tfidf_vec,
        }
        with open(embed_path, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"[OK] Saved cache: {chunk_path}, {embed_path}")

    def load(self, chunk_path, embed_path):
        """Load cached embeddings - much faster than rebuilding"""
        try:
            with open(chunk_path, "rb") as fh:
                self.chunks = pickle.load(fh)
            with open(embed_path, "rb") as fh:
                data = pickle.load(fh)
            self.embeddings = data["embeddings"]
            self._tfidf_vec = data.get("tfidf_vec")
            if data.get("backend") == "sbert" and _EMBED_BACKEND == "sbert":
                print(f"[>>] Loading embedding model '{EMBED_MODEL}' ...")
                self._model = SentenceTransformer(EMBED_MODEL)
            print(f"[OK] Loaded {len(self.chunks)} chunks from cache.")
            return True
        except Exception as exc:
            print(f"[WARN] Cache load failed ({exc}) - will rebuild from scratch.")
            return False

    def retrieve(self, query, top_k=TOP_K):
        """
        Find the top_k chunks most similar to the query.
        Uses cosine similarity between query embedding and all chunk embeddings.
        Returns chunks sorted by similarity score (highest first).
        """
        if _EMBED_BACKEND == "sbert":
            q_emb = self._model.encode([query], normalize_embeddings=True)
            sims  = (self.embeddings @ q_emb.T).ravel()
        else:
            q_vec = self._tfidf_vec.transform([query])
            sims  = _tfidf_cosine(q_vec, self.embeddings).ravel()

        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_idx:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(sims[idx])
            results.append(chunk)
        return results


# ------------------------------------------------------------------
# Google Gemini API call
# Using OpenAI-compatible endpoint so we can use standard chat format
# Note: Authorization: Bearer (not api-key like HCL AI Cafe)
#       max_tokens (snake_case, not maxTokens)
# ------------------------------------------------------------------

def call_gemini(api_key, messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model":       DEPLOYMENT,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=body, timeout=90)

        if resp.status_code == 401:
            return "[ERROR 401] Invalid or expired Gemini API key. Check at https://aistudio.google.com/app/apikey"
        if resp.status_code == 429:
            return "[ERROR 429] Rate limit hit. Please wait a moment and try again."
        resp.raise_for_status()

        data    = resp.json()
        content = data["choices"][0]["message"].get("content") or ""
        return content.strip()

    except requests.exceptions.ConnectionError:
        return "[ERROR] Cannot connect to Gemini API. Check your internet connection."
    except requests.exceptions.Timeout:
        return "[ERROR] API request timed out (>90s). Please try again."
    except (KeyError, IndexError) as exc:
        preview = resp.text[:300] if "resp" in dir() else "N/A"
        return f"[ERROR] Unexpected response format ({exc}). Raw: {preview}"
    except Exception as exc:
        return f"[ERROR] API call failed: {exc}"


# ------------------------------------------------------------------
# Prompt builders
# I format the retrieved policy excerpts as numbered context blocks
# and inject them into the user's question before sending to Gemini
# ------------------------------------------------------------------

def _format_context(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Excerpt {i} - Page {c['page']}]\n{c['text']}")
    return "\n\n".join(parts)


def build_qa_prompt(chunks, question):
    context = _format_context(chunks)
    return (
        f"[POLICY CONTEXT - SBI General Insurance]\n"
        f"{context}\n\n"
        f"{'─'*60}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"Answer using ONLY the policy excerpts above. Cite page numbers."
    )


def build_claim_prompt(chunks, scenario):
    context = _format_context(chunks)
    return (
        f"[POLICY CONTEXT - SBI General Insurance]\n"
        f"{context}\n\n"
        f"{'─'*60}\n\n"
        f"CLAIM PRE-CHECK SCENARIO:\n{scenario}\n\n"
        f"Based ONLY on the policy excerpts:\n"
        f"1. Start with verdict: [ELIGIBLE] / [POSSIBLY ELIGIBLE] / [NOT ELIGIBLE] / [NEED MORE INFORMATION]\n"
        f"2. Explain reasoning with page citations.\n"
        f"3. List any conditions, exclusions, or waiting periods.\n"
        f"4. Specify documents needed if eligible."
    )


# ------------------------------------------------------------------
# RAG Query Pipeline
# This is the core function - ties everything together:
# retrieve relevant chunks -> build prompt -> call API -> update history
# ------------------------------------------------------------------

def query(store, api_key, user_text, history, mode="qa"):
    # Step 1: Retrieve relevant policy sections
    chunks = store.retrieve(user_text, TOP_K)

    # Step 2: Build prompt with policy context injected
    if mode == "claim":
        user_content = build_claim_prompt(chunks, user_text)
    else:
        user_content = build_qa_prompt(chunks, user_text)

    # Step 3: Build messages list
    # Keep only last 6 history entries (=3 turns) to avoid huge API calls
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_content})

    # Step 4: Call Gemini
    answer = call_gemini(api_key, messages)

    # Step 5: Update history (store compact version without policy context)
    history.append({"role": "user",      "content": user_text})
    history.append({"role": "assistant", "content": answer})

    return answer, history, chunks


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------

def print_banner():
    print("\n" + "=" * 70)
    print("  POLICY & CLAIMS COPILOT")
    print("  SBI General Insurance  |  Powered by RAG + Google Gemini 2.0 Flash")
    print("=" * 70)
    print("""
  COMMANDS
  ────────────────────────────────────────────────────
  /claim    - Claim pre-check mode  (describe your scenario)
  /qa       - Q&A mode              (default)
  /sources  - Show source excerpts from the last answer
  /history  - Show conversation summary
  /clear    - Clear conversation history
  /rebuild  - Re-index the policy PDF from scratch
  /help     - Show this help
  /quit     - Exit

  EXAMPLE QUESTIONS (Q&A mode)
  ────────────────────────────────────────────────────
  "What is the waiting period for pre-existing diseases?"
  "Is dental treatment covered?"
  "What are the sub-limits for room rent charges?"
  "What documents do I need to file a hospitalisation claim?"
  "What is the claim intimation timeline?"

  EXAMPLE (Claim Pre-Check - type /claim first)
  ────────────────────────────────────────────────────
  "I was hospitalised for 3 days after a road accident.
   I have an active policy. Am I eligible to file a claim?"
""")
    print("=" * 70)


def print_response(answer, chunks, show_sources=False):
    print(f"\n{DIVIDER}")
    print("  COPILOT ANSWER")
    print(DIVIDER)

    for para in answer.split("\n"):
        stripped = para.strip()
        if stripped:
            wrapped = textwrap.fill(stripped, width=68,
                                    initial_indent="  ", subsequent_indent="  ")
            print(wrapped)
        else:
            print()

    pages  = sorted(set(c["page"] for c in chunks))
    scores = [round(c["score"], 3) for c in chunks]
    print(f"\n{DIVIDER}")
    print(f"  Source pages: {pages}   |   Similarity scores: {scores}")

    if show_sources:
        print(f"\n  {'─'*68}")
        print("  RETRIEVED POLICY EXCERPTS")
        print(f"  {'─'*68}")
        for i, c in enumerate(chunks, 1):
            print(f"\n  [{i}] Page {c['page']}  (score: {c['score']:.4f})")
            excerpt = textwrap.fill(
                c["text"][:500] + ("..." if len(c["text"]) > 500 else ""),
                width=66, initial_indent="  ", subsequent_indent="  "
            )
            print(excerpt)
    print()


# ------------------------------------------------------------------
# Build or load the RAG index
# First run: downloads PDF, extracts text, chunks, embeds (takes ~45s)
# Subsequent runs: loads from cache files (~5s)
# ------------------------------------------------------------------

def initialize(rebuild=False):
    store = VectorStore()

    if (not rebuild
            and Path(CHUNK_CACHE).exists()
            and Path(EMBED_CACHE).exists()):
        if store.load(CHUNK_CACHE, EMBED_CACHE):
            return store

    print("\n[>>] Building RAG index from scratch ...")

    if not download_pdf(PDF_URL, PDF_LOCAL):
        print("[ERROR] Cannot continue without the policy PDF. Exiting.")
        sys.exit(1)

    pages = extract_pages(PDF_LOCAL)
    if not pages:
        print("[ERROR] No text found in PDF - might be a scanned document.")
        sys.exit(1)

    chunks = chunk_pages(pages, CHUNK_WORDS, OVERLAP_WORDS)
    store.build(chunks)
    store.save(CHUNK_CACHE, EMBED_CACHE)

    return store


# ------------------------------------------------------------------
# Interactive chat loop
# Handles user input, commands, and dispatches to RAG query function
# ------------------------------------------------------------------

def run_chat(store, api_key):
    history     = []
    mode        = "qa"
    last_chunks = []

    print_banner()

    while True:
        mode_tag = "[CLAIM CHECK]" if mode == "claim" else "[Q&A]"
        try:
            user_in = input(f"\n  You {mode_tag}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break

        if not user_in:
            continue

        cmd = user_in.lower()

        if cmd in ("/quit", "/exit", "/q"):
            print("  Goodbye!")
            break

        elif cmd == "/claim":
            mode = "claim"
            print("  [>>] Claim Pre-Check mode. Describe your claim scenario in detail.")
            continue

        elif cmd == "/qa":
            mode = "qa"
            print("  [>>] Q&A mode.")
            continue

        elif cmd == "/sources":
            if last_chunks:
                print_response("", last_chunks, show_sources=True)
            else:
                print("  Ask a question first.")
            continue

        elif cmd == "/history":
            if not history:
                print("  No conversation history yet.")
            else:
                print(f"\n  Conversation ({len(history)//2} exchanges):")
                for i in range(0, len(history), 2):
                    q = history[i]["content"][:80].replace("\n", " ")
                    a = history[i+1]["content"][:80].replace("\n", " ")
                    print(f"  Q{i//2+1}: {q}")
                    print(f"  A{i//2+1}: {a}\n")
            continue

        elif cmd == "/clear":
            history = []
            print("  [>>] History cleared.")
            continue

        elif cmd == "/rebuild":
            print("  [>>] Rebuilding index ...")
            store = initialize(rebuild=True)
            print("  [OK] Index rebuilt.")
            continue

        elif cmd in ("/help", "/?"):
            print_banner()
            continue

        # normal question - run the RAG pipeline
        print(f"  [>>] Searching policy sections ...")
        t0 = time.time()
        answer, history, chunks = query(store, api_key, user_in, history, mode=mode)
        elapsed = time.time() - t0

        last_chunks = chunks
        print_response(answer, chunks)
        print(f"  Response time: {elapsed:.1f}s")


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("  Initializing Policy & Claims Copilot ...")
    print("=" * 70)

    # Check environment variable first, then prompt user
    # Can skip prompt by running: $env:GEMINI_API_KEY = "your-key"
    # Get a free key at: https://aistudio.google.com/app/apikey
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaXXXXXXXXXXXXXXXXXXXXXX").strip()

    if not api_key:
        print("\n  Enter your Google Gemini API key.")
        print("  (Free key at: https://aistudio.google.com/app/apikey)")
        print("  (Or set GEMINI_API_KEY env variable to skip this prompt)")
        try:
            api_key = getpass.getpass("  API Key: ").strip()
        except Exception:
            api_key = input("  API Key: ").strip()

    if not api_key:
        print("[ERROR] API key required.")
        sys.exit(1)

    # Quick test to check key works before building the index
    print("\n  [>>] Checking API key ...")
    test_resp = call_gemini(
        api_key,
        [{"role": "user", "content": "Reply with the single word: ready"}],
        max_tokens=50, temperature=0
    )
    if test_resp.startswith("[ERROR"):
        print(f"  {test_resp}")
        print("  Check your key at https://aistudio.google.com/app/apikey")
        sys.exit(1)
    print(f"  [OK] API working. Response: '{test_resp}'")

    # Build or load the RAG index
    store = initialize()

    # Start the chat
    run_chat(store, api_key)


if __name__ == "__main__":
    main()
