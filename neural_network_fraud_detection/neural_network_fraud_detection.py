# Assignment 3 - Visualizing Neural Network Architecture
# Use Case: Fraud Detection for Credit Card Transactions
#
# This draws the architecture of a neural network that would classify
# transactions as fraudulent or genuine. I designed the architecture
# as 6->8->6->4->1 after reading about funnel designs.
# Note: this is just the visualization, not an actual trained model.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D


# ------------------------------------------------------------------
# Network Architecture Definition
# I chose these layer sizes based on what makes sense for 6 inputs:
# - Start slightly wider to allow feature combinations
# - Then funnel down to compress into fraud probability
# ------------------------------------------------------------------

LAYERS = [
    {
        "name":       "Input Layer",
        "neurons":    6,
        "activation": "Linear\n(no activation)",
        "color":      "#AED6F1",
        "edge_color": "#2980B9",
        "label":      "x",
    },
    {
        "name":       "Hidden Layer 1",
        "neurons":    8,
        "activation": "ReLU",
        "color":      "#A9DFBF",
        "edge_color": "#1E8449",
        "label":      "h1",
    },
    {
        "name":       "Hidden Layer 2",
        "neurons":    6,
        "activation": "ReLU",
        "color":      "#A9DFBF",
        "edge_color": "#1E8449",
        "label":      "h2",
    },
    {
        "name":       "Hidden Layer 3",
        "neurons":    4,
        "activation": "ReLU",
        "color":      "#A9DFBF",
        "edge_color": "#1E8449",
        "label":      "h3",
    },
    {
        "name":       "Output Layer",
        "neurons":    1,
        "activation": "Sigmoid\nP(Fraud) in [0,1]",
        "color":      "#F9E79F",
        "edge_color": "#B7950B",
        "label":      "y^",
    },
]

# labels for the 6 input neurons
INPUT_FEATURE_LABELS = [
    "TransactionAmount",
    "TransactionTime",
    "MerchantCategory",
    "CustomerAge",
    "AccountBalance",
    "NumberOfTransactions\nToday",
]


# ------------------------------------------------------------------
# Layout - figure out where each neuron goes on the canvas
# x-axis = left to right across layers
# y-axis = neurons spread vertically within each layer, centered at 0
# ------------------------------------------------------------------

N_LAYERS    = len(LAYERS)
MAX_NEURONS = max(l["neurons"] for l in LAYERS)   # 8 neurons in H1
H_SPACING   = 2.8
NEURON_R    = 0.32
V_SCALE     = 1.0

layer_x = [i * H_SPACING for i in range(N_LAYERS)]

def neuron_y_positions(n_neurons, v_scale=V_SCALE):
    """space neurons evenly, centered at y=0"""
    if n_neurons == 1:
        return [0.0]
    spacing = (MAX_NEURONS - 1) / (n_neurons - 1) * v_scale
    top = (n_neurons - 1) / 2 * spacing
    return [top - i * spacing for i in range(n_neurons)]

# precompute (x, y) for every neuron
positions = []
for li, layer in enumerate(LAYERS):
    ys = neuron_y_positions(layer["neurons"])
    positions.append([(layer_x[li], y) for y in ys])


# ------------------------------------------------------------------
# Create the figure
# Using equal aspect ratio so circles look like circles not ovals
# ------------------------------------------------------------------

FIG_W, FIG_H = 20, 11
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_aspect("equal")
ax.axis("off")

fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

# title
ax.text(
    (layer_x[0] + layer_x[-1]) / 2, MAX_NEURONS * V_SCALE * 0.58 + 0.6,
    "Neural Network Architecture — Credit Card Fraud Detection",
    ha="center", va="bottom", fontsize=16, fontweight="bold", color="#1A252F",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB", edgecolor="#2980B9", lw=1.5)
)
ax.text(
    (layer_x[0] + layer_x[-1]) / 2, MAX_NEURONS * V_SCALE * 0.58 - 0.1,
    "Architecture: 6 -> 8 -> 6 -> 4 -> 1   |   Activations: ReLU (hidden), Sigmoid (output)",
    ha="center", va="bottom", fontsize=10, color="#555555", style="italic"
)


# ------------------------------------------------------------------
# Step 1: Draw connection lines first (so neurons appear on top)
# Every neuron in layer i connects to every neuron in layer i+1
# These are the "weights" that the network learns during training
# ------------------------------------------------------------------

for li in range(N_LAYERS - 1):
    src_positions = positions[li]
    dst_positions = positions[li + 1]
    for (x1, y1) in src_positions:
        for (x2, y2) in dst_positions:
            line = Line2D(
                [x1 + NEURON_R, x2 - NEURON_R],
                [y1, y2],
                color="#CCCCCC", linewidth=0.5, alpha=0.55, zorder=1
            )
            ax.add_line(line)


# ------------------------------------------------------------------
# Step 2: Draw neurons as circles
# Color tells you the layer type: blue=input, green=hidden, yellow=output
# ------------------------------------------------------------------

SUBSCRIPTS = "0123456789"

def sub(n):
    subs = str.maketrans("0123456789", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")
    return str(n).translate(subs)

for li, layer in enumerate(LAYERS):
    for ni, (cx, cy) in enumerate(positions[li]):
        circle = plt.Circle(
            (cx, cy), NEURON_R,
            color=layer["color"],
            ec=layer["edge_color"],
            linewidth=1.8,
            zorder=3
        )
        ax.add_patch(circle)

        subscript = sub(ni + 1)
        inner_label = f"{layer['label']}{subscript}"
        ax.text(
            cx, cy, inner_label,
            ha="center", va="center",
            fontsize=8, fontweight="bold",
            color="#1A252F", zorder=4
        )


# ------------------------------------------------------------------
# Step 3: Add feature labels on the left side of input neurons
# ------------------------------------------------------------------

for ni, (ix, iy) in enumerate(positions[0]):
    label = INPUT_FEATURE_LABELS[ni]
    ax.annotate(
        label,
        xy=(ix - NEURON_R, iy),
        xytext=(ix - NEURON_R - 1.5, iy),
        ha="right", va="center",
        fontsize=8.5, color="#1A252F",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#D6EAF8",
                  edgecolor="#2980B9", linewidth=1.0),
        arrowprops=dict(arrowstyle="->", color="#2980B9", lw=1.3),
        zorder=5
    )


# ------------------------------------------------------------------
# Step 4: Label the output neuron with P(Fraud) and decision rule
# ------------------------------------------------------------------

ox, oy = positions[-1][0]

ax.annotate(
    "P(Fraud)\nin [0, 1]",
    xy=(ox + NEURON_R, oy),
    xytext=(ox + NEURON_R + 0.5, oy + 0.55),
    ha="left", va="center",
    fontsize=9, color="#7D6608", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF9E7", edgecolor="#B7950B", linewidth=1.2),
    arrowprops=dict(arrowstyle="->", color="#B7950B", lw=1.3),
    zorder=5
)

ax.annotate(
    "  Decision Rule:\n"
    "  P(Fraud) >= 0.5  =>  [FRAUD]\n"
    "  P(Fraud)  < 0.5  =>  [GENUINE]",
    xy=(ox + NEURON_R, oy),
    xytext=(ox + NEURON_R + 0.5, oy - 0.85),
    ha="left", va="center",
    fontsize=8.5, color="#1A252F",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FDEDEC", edgecolor="#C0392B", linewidth=1.2),
    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.3),
    zorder=5
)


# ------------------------------------------------------------------
# Step 5: Add header boxes above each layer
# Shows layer name, neuron count, and activation function
# ------------------------------------------------------------------

HEADER_Y = MAX_NEURONS * V_SCALE * 0.5 + 0.55

for li, layer in enumerate(LAYERS):
    cx = layer_x[li]
    n  = layer["neurons"]

    header_text = (
        f"{layer['name']}\n"
        f"{n} neuron{'s' if n > 1 else ''}\n"
        f"Act: {layer['activation']}"
    )

    ax.text(
        cx, HEADER_Y, header_text,
        ha="center", va="bottom",
        fontsize=8, color="#1A252F",
        multialignment="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=layer["color"],
                  edgecolor=layer["edge_color"], linewidth=1.5),
        zorder=5
    )

    top_neuron_y = positions[li][0][1] + NEURON_R
    ax.plot([cx, cx], [HEADER_Y - 0.25, top_neuron_y + 0.05],
            color=layer["edge_color"], lw=0.8, ls="--", alpha=0.5, zorder=2)


# ------------------------------------------------------------------
# Step 6: Show parameter counts between layers
# W = number of weights (connections between layers)
# b = number of biases (one per neuron in destination layer)
# Total = W + b
# I calculated these manually to verify: 56+54+28+5 = 143 total
# ------------------------------------------------------------------

WEIGHT_BADGE_Y = -(MAX_NEURONS * V_SCALE * 0.5) - 0.55

for li in range(N_LAYERS - 1):
    n_src = LAYERS[li]["neurons"]
    n_dst = LAYERS[li + 1]["neurons"]
    n_w   = n_src * n_dst
    n_b   = n_dst
    total_here = n_w + n_b

    mid_x = (layer_x[li] + layer_x[li + 1]) / 2

    ax.text(
        mid_x, WEIGHT_BADGE_Y,
        f"W: {n_w} + b: {n_b}\n= {total_here} params",
        ha="center", va="top",
        fontsize=8, color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EAEDED",
                  edgecolor="#999999", linewidth=0.8),
        zorder=5
    )

# total params badge
total_params = sum(
    LAYERS[i]["neurons"] * LAYERS[i + 1]["neurons"] + LAYERS[i + 1]["neurons"]
    for i in range(N_LAYERS - 1)
)
ax.text(
    (layer_x[0] + layer_x[-1]) / 2, WEIGHT_BADGE_Y - 0.55,
    f"Total Trainable Parameters: {total_params}",
    ha="center", va="top",
    fontsize=10, fontweight="bold", color="#1A252F",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB",
              edgecolor="#2980B9", linewidth=1.5),
    zorder=5
)


# ------------------------------------------------------------------
# Step 7: Forward pass arrow at the bottom
# shows data flows left to right through the network
# ------------------------------------------------------------------

ARROW_Y = WEIGHT_BADGE_Y - 1.35

ax.annotate(
    "",
    xy=(layer_x[-1], ARROW_Y),
    xytext=(layer_x[0], ARROW_Y),
    arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=2.5, connectionstyle="arc3,rad=0.0"),
    zorder=4
)
ax.text(
    (layer_x[0] + layer_x[-1]) / 2, ARROW_Y - 0.15,
    "Forward Pass  ->  Feature Extraction  ->  Pattern Recognition  ->  Fraud Probability",
    ha="center", va="top",
    fontsize=9, color="#2C3E50", style="italic"
)


# ------------------------------------------------------------------
# Legend
# ------------------------------------------------------------------

legend_handles = [
    mpatches.Patch(color="#AED6F1", ec="#2980B9", label="Input Neuron  (raw feature value)"),
    mpatches.Patch(color="#A9DFBF", ec="#1E8449", label="Hidden Neuron (learned abstraction)"),
    mpatches.Patch(color="#F9E79F", ec="#B7950B", label="Output Neuron (fraud probability)"),
    Line2D([0], [0], color="#CCCCCC", lw=1.5, label="Weighted Connection (learnable weight)"),
]

ax.legend(
    handles=legend_handles,
    loc="lower right",
    fontsize=8.5,
    framealpha=0.92,
    edgecolor="#AAAAAA",
    facecolor="#FDFEFE",
    title="Legend",
    title_fontsize=9,
)


# ------------------------------------------------------------------
# Set axis limits and save
# ------------------------------------------------------------------

x_pad = 2.8
y_pad = 1.0
all_x = [x for li in positions for (x, _) in li]
all_y = [y for li in positions for (_, y) in li]

ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
ax.set_ylim(min(all_y) - 2.8, HEADER_Y + 1.0)

plt.tight_layout()
plt.savefig("neural_network_architecture.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved -> neural_network_architecture.png")
plt.show()


# ------------------------------------------------------------------
# Print architecture summary to console
# ------------------------------------------------------------------

print("\nNEURAL NETWORK ARCHITECTURE SUMMARY - FRAUD DETECTION")
print("-" * 60)

print("""
Input Features (6 neurons):
  1. TransactionAmount        - value of the transaction
  2. TransactionTime          - time (seconds since midnight)
  3. MerchantCategory         - encoded merchant category
  4. CustomerAge              - age of cardholder
  5. AccountBalance           - current balance
  6. NumberOfTransactionsToday- transactions today by this customer

Target: Fraud (1) or Genuine (0) - Binary Classification
""")

print(f"  {'Layer':<22} {'Neurons':>8} {'Activation':>16} {'Params':>10}")
print(f"  {'-'*60}")

prev_n = 6
total_p = 0
rows = [("Input Layer", 6, "Linear (pass-through)", "---")]
for li in range(1, N_LAYERS):
    n    = LAYERS[li]["neurons"]
    act  = LAYERS[li]["activation"].replace("\n", " ")
    p    = prev_n * n + n
    total_p += p
    rows.append((LAYERS[li]["name"], n, act, str(p)))
    prev_n = n

for (name, n, act, p) in rows:
    print(f"  {name:<22} {n:>8} {act:>16} {p:>10}")

print(f"  {'-'*60}")
print(f"  {'TOTAL PARAMS':>47} {total_p:>10}")

print(f"""
Design notes:
- 6 inputs: one per feature (raw values, would be standardized before training)
- Hidden Layer 1 (8): wider than input to capture feature combinations
  e.g., high amount + unusual time could indicate fraud
- Funnel shape (8->6->4): compresses features into abstract patterns
  each layer learns more complex representations of fraud
- ReLU activation: f(x) = max(0,x) - adds non-linearity
  without non-linearity the whole network would just be linear!
- Single sigmoid output: gives P(Fraud) between 0 and 1
  if >= 0.5 -> FLAG as FRAUD, else -> GENUINE
- Total 143 trainable parameters (weights + biases)

This is only the architecture diagram - to actually train this
network we would need real labeled transaction data and a
framework like PyTorch or TensorFlow/Keras.
""")

print("Done!")
