# Assignment 2 - KNN Classification for Loan Default Prediction
# 
# Goal: predict if a customer will default on their loan before approving it
# I used KNN because it's a simple distance-based method that I understood well
# Also added a Decision Tree comparison at the end

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     LeaveOneOut, cross_val_predict)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 13, "axes.labelsize": 11})


# ------------------------------------------------------------------
# Dataset - 10 loan applicants
# Mix of numeric and categorical features
# Default = 1 means the person didn't repay the loan
# ------------------------------------------------------------------

data = {
    "Age":            [28,   45,   35,   50,   30,   42,   26,   48,   38,   55  ],
    "AnnualIncome":   [6.5,  12.0, 8.0,  15.0, 7.0,  10.0, 5.5,  14.0, 9.0,  16.0],
    "CreditScore":    [720,  680,  750,  640,  710,  660,  730,  650,  700,  620 ],
    "LoanAmount":     [5,    10,   6,    12,   5,    9,    4,    11,   7,    13  ],
    "LoanTerm":       [5,    10,   7,    15,   5,    10,   4,    12,   8,    15  ],
    "EmploymentType": ["Salaried","Self-Employed","Salaried","Self-Employed",
                       "Salaried","Salaried","Salaried","Self-Employed",
                       "Salaried","Self-Employed"],
    "Default":        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
}

df = pd.DataFrame(data)

print("LOAN DEFAULT PREDICTION - KNN CLASSIFIER")
print("-" * 50)


# ------------------------------------------------------------------
# Step 1: EDA
# quick look at the data to spot anything unusual
# ------------------------------------------------------------------

print("\nDataset:")
print(df.to_string(index=False))
print(f"\nShape: {df.shape}")
print("\nStats:")
print(df.describe().round(2).to_string())
print(f"\nMissing: {df.isnull().sum().sum()}")


# ------------------------------------------------------------------
# Step 2: Encode categorical column
# KNN works on distances so we need numbers, not text
# "Salaried" -> 0, "Self-Employed" -> 1
# This works fine for 2 categories - for more than 2 would use one-hot
# ------------------------------------------------------------------

df["EmpType_enc"] = df["EmploymentType"].map({"Salaried": 0, "Self-Employed": 1})

print("\nLabel Encoding for EmploymentType:")
enc_show = df[["EmploymentType", "EmpType_enc"]].drop_duplicates().sort_values("EmpType_enc")
print(enc_show.to_string(index=False))
print("\nWhy encode? KNN uses Euclidean distance: sqrt(sum((xi - xj)^2))")
print("Can't do 'Salaried' - 'Self-Employed' mathematically, need 0 and 1")


# ------------------------------------------------------------------
# Step 3: Check class balance
# Important to know if data is balanced or one class dominates
# in real banking, defaults are rare (~5-20%) which causes issues
# ------------------------------------------------------------------

print("\nClass Distribution:")
class_counts = df["Default"].value_counts().sort_index()
for cls, cnt in class_counts.items():
    label = "No Default (0)" if cls == 0 else "Default     (1)"
    bar = "#" * (cnt * 6)
    print(f"  {label}: {cnt}  {bar}")

ratio = class_counts[0] / class_counts[1]
print(f"\n  Ratio 0:1 = {class_counts[0]}:{class_counts[1]} - {'Balanced' if 0.8 < ratio < 1.2 else 'Imbalanced'}")
print("""
  Note: real bank datasets are usually very imbalanced.
  A model that just says "No Default" for everyone would be 90%+ accurate
  but totally useless. Always check Recall for the Default class!
  Real fixes: SMOTE oversampling, class_weight='balanced' etc.
""")


# ------------------------------------------------------------------
# Step 4: Prepare features
# ------------------------------------------------------------------

feature_cols = ["Age", "AnnualIncome", "CreditScore", "LoanAmount", "LoanTerm", "EmpType_enc"]
X = df[feature_cols]
y = df["Default"]

print("Feature Matrix X:")
print(X.to_string(index=False))
print(f"\nTarget y: {list(y.values)}  (0=No Default, 1=Default)")


# ------------------------------------------------------------------
# Step 5: Why scaling matters for KNN
# This is something I had to think about carefully
# CreditScore is in hundreds (620-750 range)
# AnnualIncome is in single digits (5.5-16 range)
# Without scaling, credit score would completely dominate the distance
# ------------------------------------------------------------------

print("\nWhy Scaling is Critical for KNN:")
print("-" * 40)
person_a = np.array([720, 6.5])   # CreditScore, AnnualIncome
person_b = np.array([680, 12.0])
person_c = np.array([715, 14.0])

# unscaled distance
def eucl(a, b):
    return np.sqrt(np.sum((a - b)**2))

d_ab_raw = eucl(person_a, person_b)
d_ac_raw = eucl(person_a, person_c)
print(f"  Without scaling: dist(A,B) = {d_ab_raw:.2f}, dist(A,C) = {d_ac_raw:.2f}")
print(f"  -> Credit score difference of 40 vs income difference of 7.5...")
print(f"  -> CreditScore dominates because its scale is much larger")

# now scaled
from sklearn.preprocessing import StandardScaler as _SS
_ss = _SS()
scaled_pts = _ss.fit_transform(np.array([[720, 6.5], [680, 12.0], [715, 14.0]]))
d_ab_sc = eucl(scaled_pts[0], scaled_pts[1])
d_ac_sc = eucl(scaled_pts[0], scaled_pts[2])
print(f"\n  After scaling:   dist(A,B) = {d_ab_sc:.4f}, dist(A,C) = {d_ac_sc:.4f}")
print(f"  -> Now each feature contributes fairly based on its variation")
print(f"  -> CreditScore dominates unscaled: {d_ab_raw/d_ab_sc:.1f}x amplified without scaling")


# ------------------------------------------------------------------
# Step 6: Scale the data
# ------------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

print("\nScaled Feature Matrix:")
print(X_scaled_df.round(3).to_string(index=False))


# ------------------------------------------------------------------
# Step 7: Find best K using LOOCV
# I tried K from 1 to 7 and measured accuracy, F1, recall
# With 10 samples, LOOCV is better than 80/20 split
# ------------------------------------------------------------------

k_values = range(1, 8)
cv_accuracy, cv_f1, cv_recall = [], [], []
loo = LeaveOneOut()

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_accuracy.append(cross_val_score(knn, X_scaled, y, cv=loo, scoring="accuracy").mean())
    cv_f1.append(cross_val_score(knn, X_scaled, y, cv=loo, scoring="f1").mean())
    cv_recall.append(cross_val_score(knn, X_scaled, y, cv=loo, scoring="recall").mean())

print("\nFinding Best K (LOOCV):")
print(f"  {'K':>3}  {'Accuracy':>10}  {'F1':>8}  {'Recall':>8}")
for k, acc, f1, rec in zip(k_values, cv_accuracy, cv_f1, cv_recall):
    print(f"  {k:>3}  {acc:>10.4f}  {f1:>8.4f}  {rec:>8.4f}")

best_k = k_values[np.argmax(cv_accuracy)]
print(f"\nBest K = {best_k} (highest LOOCV accuracy)")


# ------------------------------------------------------------------
# Step 8: Train final KNN model with best K
# Using full dataset with LOOCV predictions for metrics
# ------------------------------------------------------------------

knn_best = KNeighborsClassifier(n_neighbors=best_k)
y_pred_loocv = cross_val_predict(knn_best, X_scaled, y, cv=loo)

acc   = accuracy_score(y, y_pred_loocv)
prec  = precision_score(y, y_pred_loocv)
rec   = recall_score(y, y_pred_loocv)
f1    = f1_score(y, y_pred_loocv)
try:
    auc = roc_auc_score(y, y_pred_loocv)
except Exception:
    auc = 0.0

print(f"\nKNN (K={best_k}) Performance (LOOCV):")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")

cm = confusion_matrix(y, y_pred_loocv)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")


# ------------------------------------------------------------------
# Step 9: Compare with Decision Tree
# Just to see how KNN stacks up against another classifier
# ------------------------------------------------------------------

dt = DecisionTreeClassifier(random_state=42)
y_pred_dt = cross_val_predict(dt, X_scaled, y, cv=loo)

dt_acc   = accuracy_score(y, y_pred_dt)
dt_prec  = precision_score(y, y_pred_dt)
dt_rec   = recall_score(y, y_pred_dt)
dt_f1    = f1_score(y, y_pred_dt)
try:
    dt_auc = roc_auc_score(y, y_pred_dt)
except Exception:
    dt_auc = 0.0

print(f"\nComparison - KNN vs Decision Tree (LOOCV):")
print(f"  {'Metric':<12}  {'KNN K='+str(best_k):>10}  {'DecTree':>10}")
print(f"  {'-------':<12}  {'------':>10}  {'-------':>10}")
print(f"  {'Accuracy':<12}  {acc:>10.4f}  {dt_acc:>10.4f}")
print(f"  {'Precision':<12}  {prec:>10.4f}  {dt_prec:>10.4f}")
print(f"  {'Recall':<12}  {rec:>10.4f}  {dt_rec:>10.4f}")
print(f"  {'F1 Score':<12}  {f1:>10.4f}  {dt_f1:>10.4f}")
print(f"  {'ROC-AUC':<12}  {auc:>10.4f}  {dt_auc:>10.4f}")


# ------------------------------------------------------------------
# Step 10: Predict for a new applicant
# ------------------------------------------------------------------

new_applicant = pd.DataFrame(
    [[35, 7.5, 670, 8, 10, 1]],
    columns=feature_cols
)
new_scaled = scaler.transform(new_applicant)

knn_best.fit(X_scaled, y)
pred_class = knn_best.predict(new_scaled)[0]
pred_proba = knn_best.predict_proba(new_scaled)[0]

print(f"\nNew Applicant Prediction:")
print(f"  Age=35, Income=7.5L, CreditScore=670, Loan=8L, Term=10yr, Self-Employed")
print(f"  Predicted class: {'DEFAULT RISK' if pred_class == 1 else 'LIKELY REPAY'}")
print(f"  Probability: No Default={pred_proba[0]:.3f}, Default={pred_proba[1]:.3f}")


# ------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------

fig = plt.figure(figsize=(18, 10))
fig.suptitle("Loan Default Prediction — KNN Classification Analysis",
             fontsize=15, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# Plot 1: Finding best K
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(k_values, cv_accuracy, "bo-", linewidth=2, markersize=7, label="CV Accuracy")
ax1.plot(k_values, cv_f1,       "g^--", linewidth=1.5, markersize=7, label="CV F1")
ax1.plot(k_values, cv_recall,   "rD--", linewidth=1.5, markersize=7, label="CV Recall")
ax1.axvline(best_k, color="red", linestyle=":", linewidth=1.5, label=f"Optimal K={best_k}")
ax1.set_xlabel("K (Number of Neighbours)")
ax1.set_ylabel("Score")
ax1.set_title(f"Finding Optimal K\n(LOOCV: Accuracy, F1, Recall)", fontweight="bold")
ax1.legend(fontsize=8)
ax1.set_ylim(0, 1.1)
ax1.set_xticks(list(k_values))

# Plot 2: Confusion Matrix
ax2 = fig.add_subplot(gs[0, 1])
labels = ["TN", "FP", "FN", "TP"]
cm_flat = cm.ravel()
colors_cm = ["#1a5276", "#a93226", "#a93226", "#1a5276"]
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 2)
ax2.set_xticks([0.5, 1.5])
ax2.set_yticks([0.5, 1.5])
ax2.set_xticklabels(["Predicted\nNo Default", "Predicted\nDefault"])
ax2.set_yticklabels(["Actual\nDefault", "Actual\nNo Default"])
for idx, (label, val, color) in enumerate(zip(labels, [cm[0,0], cm[0,1], cm[1,0], cm[1,1]], colors_cm)):
    r, c = divmod(idx, 2)
    ax2.add_patch(plt.Rectangle((c, 1-r), 1, 1, color=color, alpha=0.85))
    ax2.text(c+0.5, 1.5-r, f"{label}\n{val}", ha="center", va="center",
             fontsize=14, fontweight="bold", color="white")
ax2.set_title(f"Confusion Matrix\n(LOOCV, KNN K={best_k})", fontweight="bold")

# Plot 3: ROC Curves
ax3 = fig.add_subplot(gs[0, 2])
knn_best.fit(X_scaled, y)
try:
    fpr_knn, tpr_knn, _ = roc_curve(y, cross_val_predict(knn_best, X_scaled, y, cv=loo, method="predict_proba")[:, 1])
    ax3.plot(fpr_knn, tpr_knn, "bo-", linewidth=2, label=f"KNN K={best_k} (AUC={auc:.2f})")
except Exception:
    ax3.plot([0,1], [0,1], "bo-", label=f"KNN K={best_k} (AUC={auc:.2f})")
try:
    fpr_dt, tpr_dt, _ = roc_curve(y, cross_val_predict(dt, X_scaled, y, cv=loo, method="predict_proba")[:, 1])
    ax3.plot(fpr_dt, tpr_dt, "rs-", linewidth=2, label=f"Decision Tree (AUC={dt_auc:.2f})")
except Exception:
    ax3.plot([0,1], [0,1], "rs-", label=f"Decision Tree")
ax3.plot([0,1], [0,1], "k--", linewidth=1, label="Random (AUC=0.50)")
ax3.set_xlabel("False Positive Rate  (1 - Specificity)")
ax3.set_ylabel("True Positive Rate (Recall / Sensitivity)")
ax3.set_title("ROC Curve\n(KNN vs Decision Tree)", fontweight="bold")
ax3.legend(fontsize=8)

# Plot 4: Credit Score vs Income scatter
ax4 = fig.add_subplot(gs[1, 0])
colors_scatter = {0: "green", 1: "red"}
markers_scatter = {0: "o", 1: "X"}
for cls in [0, 1]:
    mask = y == cls
    lbl = "No Default" if cls == 0 else "Default"
    ax4.scatter(df.loc[mask, "CreditScore"], df.loc[mask, "AnnualIncome"],
                c=colors_scatter[cls], marker=markers_scatter[cls],
                s=100, label=lbl, zorder=3)
    for _, row in df[mask].iterrows():
        ax4.annotate(f"Age {row['Age']}", (row["CreditScore"], row["AnnualIncome"]),
                     textcoords="offset points", xytext=(5, 3), fontsize=7)
ax4.axvline(680, color="orange", linestyle="--", linewidth=1.5, alpha=0.7)
ax4.set_xlabel("Credit Score")
ax4.set_ylabel("Annual Income (lakhs)")
ax4.set_title("Credit Score vs Income\n(by Default status)", fontweight="bold")
ax4.legend(fontsize=9)

# Plot 5: Loan Amount vs Loan Term
ax5 = fig.add_subplot(gs[1, 1])
for cls in [0, 1]:
    mask = y == cls
    lbl = "No Default" if cls == 0 else "Default"
    ax5.scatter(df.loc[mask, "LoanAmount"], df.loc[mask, "LoanTerm"],
                c=colors_scatter[cls], marker=markers_scatter[cls],
                s=100, label=lbl, zorder=3)
    for _, row in df[mask].iterrows():
        emp_short = "Sal" if row["EmploymentType"] == "Salaried" else "Sel"
        ax5.annotate(emp_short, (row["LoanAmount"], row["LoanTerm"]),
                     textcoords="offset points", xytext=(5, 3), fontsize=7)
ax5.axvline(8, color="orange", linestyle="--", linewidth=1.5, alpha=0.7)
ax5.set_xlabel("Loan Amount (lakhs)")
ax5.set_ylabel("Loan Term (years)")
ax5.set_title("Loan Amount vs Loan Term\n(by Default status)", fontweight="bold")
ax5.legend(fontsize=9)

# Plot 6: KNN vs DT metric comparison bar chart
ax6 = fig.add_subplot(gs[1, 2])
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
knn_vals = [acc, prec, rec, f1, auc]
dt_vals  = [dt_acc, dt_prec, dt_rec, dt_f1, dt_auc]
x_pos = np.arange(len(metrics))
width = 0.35
bars1 = ax6.bar(x_pos - width/2, knn_vals, width, label=f"KNN K={best_k}", color="#3498db")
bars2 = ax6.bar(x_pos + width/2, dt_vals,  width, label="Decision Tree",   color="#e67e22")
ax6.bar_label(bars1, fmt="%.2f", fontsize=7, padding=2)
ax6.bar_label(bars2, fmt="%.2f", fontsize=7, padding=2)
ax6.set_ylim(0, 1.2)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(metrics, rotation=15, fontsize=9)
ax6.set_ylabel("Score")
ax6.set_title(f"KNN vs Decision Tree\n(LOOCV Metrics Comparison)", fontweight="bold")
ax6.legend(fontsize=9)

plt.savefig("loan_default_analysis.png", bbox_inches="tight")
print("\nChart saved as loan_default_analysis.png")
plt.show()

print("\nDone!")
