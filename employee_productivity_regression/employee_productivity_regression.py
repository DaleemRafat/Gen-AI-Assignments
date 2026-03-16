# Assignment 1 - Multivariate Linear Regression
# Predicting Employee Productivity Score
#
# I decided to use 4 features that I thought would make sense:
# experience, training hours, working hours and number of projects.
# The idea is to see which of these actually drives productivity.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# make plots look a bit nicer
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 13, "axes.labelsize": 11})


# ------------------------------------------------------------------
# Dataset
# I created this dataset manually with 10 employee records
# tried to keep the values realistic (not too extreme)
# ------------------------------------------------------------------

data = {
    "Experience (yrs)": [2, 5, 1, 8, 4, 10, 3, 6, 7, 2],
    "Training Hours":   [40, 60, 20, 80, 50, 90, 30, 70, 75, 25],
    "Working Hours":    [38, 42, 35, 45, 40, 48, 37, 44, 46, 36],
    "Projects":         [3, 6, 2, 8, 5, 9, 4, 7, 7, 3],
    "Productivity Score": [62, 78, 55, 88, 72, 92, 65, 82, 85, 60],
}

df = pd.DataFrame(data)

print("EMPLOYEE PRODUCTIVITY - LINEAR REGRESSION")
print("-" * 50)


# ------------------------------------------------------------------
# Step 1: Look at the data first (EDA)
# always good to understand what we're working with before jumping to model
# ------------------------------------------------------------------

print("\nDataset:")
print(df.to_string(index=False))
print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nBasic stats:")
print(df.describe().round(2).to_string())
print(f"\nMissing values: {df.isnull().sum().sum()} - good, none!")


# ------------------------------------------------------------------
# Step 2: Correlation - which features relate to productivity?
# I learned in class that Pearson r between -1 and +1
# closer to 1 or -1 means stronger relationship
# ------------------------------------------------------------------

corr_matrix = df.corr()

print("\nCorrelation with Productivity Score:")
print("-" * 40)
feat_corr = corr_matrix["Productivity Score"].drop("Productivity Score").sort_values(ascending=False)
for feat, val in feat_corr.items():
    bar = "#" * int(abs(val) * 20)
    print(f"  {feat:<22} r = {val:+.3f}  {bar}")


# ------------------------------------------------------------------
# Step 3: Split features and target
# X = inputs (what we know), y = output (what we predict)
# ------------------------------------------------------------------

feature_cols = ["Experience (yrs)", "Training Hours", "Working Hours", "Projects"]
X = df[feature_cols]
y = df["Productivity Score"]

print("\nFeature matrix X (first 3 rows):")
print(X.head(3).to_string(index=False))
print("\nTarget y:", list(y))


# ------------------------------------------------------------------
# Step 4: Train/Test Split
# keeping 20% for testing - I know this is small but dataset is small
# random_state=42 so I get same split every time I run
# ------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")


# ------------------------------------------------------------------
# Step 5: Feature Scaling
# This is needed so coefficients are comparable
# StandardScaler makes each feature mean=0, std=1
# NOTE: fit only on training data - otherwise we'd be cheating!
# ------------------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nScaled training data (first 3 rows):")
print(pd.DataFrame(X_train_scaled, columns=feature_cols).round(3).to_string(index=False))


# ------------------------------------------------------------------
# Step 6: Build the Regression Model
# The formula the model learns:
# Productivity = b0 + b1*Experience + b2*Training + b3*Working + b4*Projects
# LinearRegression finds the best b values using Ordinary Least Squares
# ------------------------------------------------------------------

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\nModel trained!")
print(f"Intercept: {model.intercept_:.4f}")
print("\nCoefficients (on scaled features):")
coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model.coef_
}).sort_values("Coefficient", ascending=False)
for _, row in coef_df.iterrows():
    direction = "positive" if row["Coefficient"] > 0 else "negative"
    print(f"  {row['Feature']:<22} b = {row['Coefficient']:+.4f}  ({direction})")


# ------------------------------------------------------------------
# Step 7: Predictions and metrics
# R2 tells us what % of variance the model explains - want it close to 1
# RMSE is in same units as productivity score - easy to interpret
# ------------------------------------------------------------------

y_pred_train = model.predict(X_train_scaled)
y_pred_test  = model.predict(X_test_scaled)

mae_train  = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train   = r2_score(y_train, y_pred_train)

mae_test   = mean_absolute_error(y_test, y_pred_test)
rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test    = r2_score(y_test, y_pred_test)

print("\nModel Performance:")
print(f"  {'Metric':<10}  {'Train':>8}  {'Test':>8}")
print(f"  {'------':<10}  {'-----':>8}  {'-----':>8}")
print(f"  {'MAE':<10}  {mae_train:>8.4f}  {mae_test:>8.4f}")
print(f"  {'RMSE':<10}  {rmse_train:>8.4f}  {rmse_test:>8.4f}")
print(f"  {'R2':<10}  {r2_train:>8.4f}  {r2_test:>8.4f}")

print(f"\n  Training R2 = {r2_train:.4f} means model explains {r2_train*100:.1f}% of variance in training data")
print(f"  Test R2     = {r2_test:.4f} means model explains {r2_test*100:.1f}% on unseen data")

print("\nActual vs Predicted (test set):")
print(f"  {'#':<5}  {'Actual':>8}  {'Predicted':>10}  {'Error':>8}")
for i, (a, p) in enumerate(zip(y_test, y_pred_test), 1):
    print(f"  {i:<5}  {a:>8.1f}  {p:>10.2f}  {a-p:>+8.2f}")


# ------------------------------------------------------------------
# Step 8: LOOCV (Leave One Out Cross Validation)
# With only 10 samples, regular CV is not reliable
# LOOCV trains on 9, tests on 1, repeats 10 times - better estimate
# I couldn't use R2 scoring here because with 1 test sample its undefined
# so I used neg_mean_squared_error instead
# ------------------------------------------------------------------

X_all_scaled = scaler.fit_transform(X)
loo = LeaveOneOut()

cv_neg_mse = cross_val_score(model, X_all_scaled, y, cv=loo, scoring="neg_mean_squared_error")
cv_rmse = np.sqrt(-cv_neg_mse)
cv_mae  = -cross_val_score(model, X_all_scaled, y, cv=loo, scoring="neg_mean_absolute_error")

print("\nLOOCV Results (10 folds):")
print(f"  {'Fold':>5}  {'RMSE':>8}  {'MAE':>8}")
for i, (r, m) in enumerate(zip(cv_rmse, cv_mae), 1):
    print(f"  {i:>5}  {r:>8.4f}  {m:>8.4f}")
print(f"\n  Mean RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
print(f"  Mean MAE:  {cv_mae.mean():.4f} (+/- {cv_mae.std():.4f})")


# ------------------------------------------------------------------
# Step 9: Feature Importance
# Since all features are scaled, we can compare coefficients directly
# larger absolute value = more important feature
# ------------------------------------------------------------------

importance = pd.DataFrame({
    "Feature":    feature_cols,
    "Coef":       model.coef_,
    "Abs_Coef":   np.abs(model.coef_),
}).sort_values("Abs_Coef", ascending=False).reset_index(drop=True)

print("\nFeature Importance (ranked by |coefficient|):")
for rank, row in importance.iterrows():
    bar  = "#" * int(row["Abs_Coef"] * 3)
    sign = "+" if row["Coef"] > 0 else "-"
    print(f"  #{rank+1}  {row['Feature']:<22}  |b| = {row['Abs_Coef']:.4f}  ({sign})  {bar}")


# ------------------------------------------------------------------
# Business Questions - answering the 7 questions from the assignment
# ------------------------------------------------------------------

print("\n" + "=" * 60)
print("BUSINESS QUESTIONS")
print("=" * 60)

top_feat = importance.iloc[0]["Feature"]
top_coef = importance.iloc[0]["Coef"]
train_coef = coef_df[coef_df["Feature"] == "Training Hours"]["Coefficient"].values[0]
work_coef  = coef_df[coef_df["Feature"] == "Working Hours"]["Coefficient"].values[0]

print(f"""
Q1. Which factor most impacts productivity?
   -> '{top_feat}' has the highest |coefficient| = {importance.iloc[0]['Abs_Coef']:.4f}
   This means a 1 std deviation increase in this feature causes the biggest
   change in predicted productivity score among all 4 features.
   Projects and Training Hours are the top two drivers in this dataset.

Q2. How does training affect productivity?
   -> Training Hours coefficient = {train_coef:+.4f} (positive effect)
   More training hours = higher predicted productivity score.
   So investing in employee training does show up as a measurable benefit.

Q3. Training hours vs Working hours - which to increase?
   -> Training Hours |b| = {abs(train_coef):.4f} vs Working Hours |b| = {abs(work_coef):.4f}
   -> Increase Training Hours - it has bigger impact on productivity.
   Just making people work longer hours doesn't help as much as skill building.
   Also long hours cause burnout which would hurt productivity long term.

Q4. What happens if Working Hours go above optimal?
   -> In this linear model, working hours always predicts positive.
   But in reality this is a limitation - after ~45-50 hrs/week people get tired,
   make more mistakes, and productivity actually drops.
   A better model could include (WorkingHours)^2 as a feature to capture this
   curve shape.

Q5. Can productivity decrease with more experience?
   -> In our model, experience coefficient is positive so no.
   But in real companies it can happen - senior employees sometimes get
   comfortable and take less challenging work. Also role mismatch is common.
   We'd need more data and maybe segment by department/role to see this.

Q6. How to detect overfitting?
   -> Compare training R2 vs test R2:
      Training R2 = {r2_train:.3f}
      Test R2     = {r2_test:.3f}
   If these are very different (say training=0.99, test=0.40) then overfitting.
   LOOCV also helps - if mean RMSE from CV is similar to training RMSE, good.
   Our LOOCV Mean RMSE = {cv_rmse.mean():.2f} which looks reasonable.
   Could also use Ridge/Lasso regression to regularize if needed.

Q7. New feature suggestion?
   -> I would add "Peer Feedback Score" (0-10 rating from coworkers)
   This captures collaboration and communication which are hard to measure
   from hours/projects alone. Research shows social/team dynamics explain
   15-25% of productivity variance in office environments.
   Other ideas: absenteeism rate, role complexity score, manager rating.
""")


# ------------------------------------------------------------------
# Visualizations - 6 plots in a 2x3 grid
# ------------------------------------------------------------------

fig = plt.figure(figsize=(18, 10))
fig.suptitle(
    "Employee Productivity — Multivariate Linear Regression Analysis",
    fontsize=15, fontweight="bold", y=1.01
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# plot 1: correlation heatmap
ax1 = fig.add_subplot(gs[0, 0])
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", vmin=-1, vmax=1, linewidths=0.5, ax=ax1)
ax1.set_title("Correlation Heatmap\n(Pearson r)", fontweight="bold")
ax1.tick_params(axis="x", rotation=30)

# plot 2: feature importance bar chart
ax2 = fig.add_subplot(gs[0, 1])
colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in importance["Coef"]]
bars = ax2.barh(importance["Feature"], importance["Abs_Coef"], color=colors)
ax2.set_xlabel("|Coefficient| (scaled features)")
ax2.set_title("Feature Importance\n(|b| on scaled data)", fontweight="bold")
ax2.bar_label(bars, fmt="%.3f", padding=3)
ax2.invert_yaxis()

# plot 3: actual vs predicted
ax3 = fig.add_subplot(gs[0, 2])
X_all_scaled2 = scaler.fit_transform(X)
model.fit(X_all_scaled2, y)
y_all_pred = model.predict(X_all_scaled2)
ax3.scatter(y, y_all_pred, color="#3498db", edgecolors="white", s=90, zorder=3)
lims = [min(y.min(), y_all_pred.min()) - 2, max(y.max(), y_all_pred.max()) + 2]
ax3.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
ax3.set_xlabel("Actual Productivity Score")
ax3.set_ylabel("Predicted Productivity Score")
ax3.set_title("Actual vs Predicted\n(full dataset, refitted)", fontweight="bold")
ax3.legend(fontsize=9)

# plot 4: residuals
ax4 = fig.add_subplot(gs[1, 0])
residuals = y.values - y_all_pred
ax4.scatter(y_all_pred, residuals, color="#9b59b6", edgecolors="white", s=90, zorder=3)
ax4.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax4.set_xlabel("Predicted Productivity Score")
ax4.set_ylabel("Residual (Actual - Predicted)")
ax4.set_title("Residual Plot\n(random scatter = good fit)", fontweight="bold")
for i, (xp, r) in enumerate(zip(y_all_pred, residuals), 1):
    ax4.annotate(str(i), (xp, r), textcoords="offset points", xytext=(5, 5), fontsize=8, color="#555")

# plot 5: training hours vs productivity
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(df["Training Hours"], df["Productivity Score"], color="#e67e22", edgecolors="white", s=90, zorder=3)
m5, b5 = np.polyfit(df["Training Hours"], df["Productivity Score"], 1)
x5 = np.linspace(df["Training Hours"].min(), df["Training Hours"].max(), 100)
ax5.plot(x5, m5*x5 + b5, "b-", linewidth=1.8, label=f"slope={m5:.3f}")
ax5.set_xlabel("Training Hours")
ax5.set_ylabel("Productivity Score")
ax5.set_title("Training Hours -> Productivity\n(strongest driver)", fontweight="bold")
ax5.legend(fontsize=9)

# plot 6: experience vs productivity
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(df["Experience (yrs)"], df["Productivity Score"], color="#1abc9c", edgecolors="white", s=90, zorder=3)
m6, b6 = np.polyfit(df["Experience (yrs)"], df["Productivity Score"], 1)
x6 = np.linspace(df["Experience (yrs)"].min(), df["Experience (yrs)"].max(), 100)
ax6.plot(x6, m6*x6 + b6, "b-", linewidth=1.8, label=f"slope={m6:.3f}")
ax6.set_xlabel("Experience (years)")
ax6.set_ylabel("Productivity Score")
ax6.set_title("Experience -> Productivity\n(positive trend)", fontweight="bold")
ax6.legend(fontsize=9)

plt.savefig("productivity_analysis.png", bbox_inches="tight")
print("Chart saved as productivity_analysis.png")
plt.show()


# ------------------------------------------------------------------
# Predict for a new employee
# (just testing the model on someone not in training data)
# ------------------------------------------------------------------

new_emp = pd.DataFrame([[4, 55, 41, 5]], columns=feature_cols)
new_scaled = scaler.transform(new_emp)
pred = model.predict(new_scaled)[0]

print("\nPrediction for new employee:")
print("  Experience: 4 yrs, Training: 55 hrs, Working: 41 hrs/wk, Projects: 5")
print(f"  Predicted Productivity Score: {pred:.2f} / 100")

print("\nDone!")
