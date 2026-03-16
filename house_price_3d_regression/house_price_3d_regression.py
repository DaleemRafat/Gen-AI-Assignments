# Assignment 4 - 3D Visualization of House Price Regression Model
#
# Task: predict house price using Area (sq ft) and Number of Bedrooms
# and show it as a 3D plot where the regression model is a plane in 3D space
#
# I thought this was really cool - you can actually see the regression
# plane floating through the data points in 3D!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
# Generate synthetic dataset
# Using a formula I made up: Price = 120*Area + 18000*Bedrooms + 30000
# then adding some random noise to make it realistic
# random seed = 42 so I get same data every run
# ------------------------------------------------------------------

np.random.seed(42)
N = 100

area     = np.random.randint(600, 4200, N).astype(float)
bedrooms = np.random.randint(1, 7, N).astype(float)
noise    = np.random.normal(0, 15_000, N)

price    = 120 * area + 18_000 * bedrooms + 30_000 + noise

df = pd.DataFrame({
    "Area_sqft":  area,
    "Bedrooms":   bedrooms,
    "HousePrice": price.round(2),
})

print("HOUSE PRICE PREDICTION - 3D REGRESSION VISUALIZATION")
print("-" * 55)

print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))

print("\nDataset summary:")
print(df.describe().round(2).to_string())

print("\nCorrelation with House Price:")
corr = df.corr(numeric_only=True)["HousePrice"].drop("HousePrice")
for feat, val in corr.items():
    bar = "#" * int(abs(val) * 30)
    print(f"  {feat:<15} r = {val:+.4f}  |{bar}|")


# ------------------------------------------------------------------
# Prepare features and split
# X has 2 columns: Area and Bedrooms
# y is the price
# ------------------------------------------------------------------

X = df[["Area_sqft", "Bedrooms"]].values
y = df["HousePrice"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\nTrain: {len(X_train)} samples, Test: {len(X_test)} samples")


# ------------------------------------------------------------------
# Scale features
# Area is in hundreds/thousands, bedrooms in 1-6 range
# Need to scale or Area would dominate - StandardScaler handles this
# Remember: fit on train only, then transform both
# ------------------------------------------------------------------

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTraining feature means: {scaler.mean_}")
print(f"Training feature stds:  {np.sqrt(scaler.var_)}")


# ------------------------------------------------------------------
# Linear Regression model
# The model learns: Price = b0 + b1*Area_scaled + b2*Bedrooms_scaled
# In 3D this is a flat PLANE, not a line like in 2D
# ------------------------------------------------------------------

model = LinearRegression()
model.fit(X_train_sc, y_train)

b0 = model.intercept_
b1, b2 = model.coef_

print(f"\nRegression Plane (scaled features):")
print(f"  Price = {b0:,.2f} + {b1:,.2f} * Area (scaled) + {b2:,.2f} * Bedrooms (scaled)")


# ------------------------------------------------------------------
# Evaluate on test set
# ------------------------------------------------------------------

y_pred_train = model.predict(X_train_sc)
y_pred_test  = model.predict(X_test_sc)

r2_train  = r2_score(y_train, y_pred_train)
r2_test   = r2_score(y_test,  y_pred_test)
mae_test  = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nModel Performance:")
print(f"  {'Metric':<25} {'Train':>12} {'Test':>12}")
print(f"  {'-'*49}")
print(f"  {'R2':<25} {r2_train:>12.4f} {r2_test:>12.4f}")
print(f"  {'MAE (USD)':<25} {'---':>12} {mae_test:>12,.2f}")
print(f"  {'RMSE (USD)':<25} {'---':>12} {rmse_test:>12,.2f}")

residuals = y_test - y_pred_test


# ------------------------------------------------------------------
# Build the regression plane surface for plotting
# I create a grid of Area and Bedroom values, predict price for each
# then use plot_surface to draw the plane
# ------------------------------------------------------------------

area_range    = np.linspace(df["Area_sqft"].min(), df["Area_sqft"].max(), 50)
bedroom_range = np.linspace(df["Bedrooms"].min(),  df["Bedrooms"].max(),  6)

GRID_A, GRID_B = np.meshgrid(area_range, bedroom_range)
grid_flat      = np.column_stack([GRID_A.ravel(), GRID_B.ravel()])
grid_flat_sc   = scaler.transform(grid_flat)
GRID_PRICE     = model.predict(grid_flat_sc).reshape(GRID_A.shape)

# color map for scatter points (red = cheap, green = expensive)
norm = mcolors.Normalize(vmin=df["HousePrice"].min(), vmax=df["HousePrice"].max())
cmap = plt.colormaps["RdYlGn"]
scatter_colors = cmap(norm(df["HousePrice"].values))


# ------------------------------------------------------------------
# Create 4-panel 3D figure
# Panel 1: main 3D view with regression plane and data points
# Panel 2: bird's eye top-down view
# Panel 3: side view showing price vs area
# Panel 4: residuals plot - shows prediction errors as colored lines
# ------------------------------------------------------------------

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#F0F3F4")
fig.suptitle(
    "3D Regression Model — House Price Prediction\n"
    f"Area (sq ft) + Bedrooms  |  R² = {r2_test:.4f}  |  RMSE = ${rmse_test:,.0f}",
    fontsize=15, fontweight="bold", y=0.98, color="#1A252F"
)

PLANE_ALPHA = 0.35
PLANE_COLOR = "#5DADE2"


# Panel 1: main 3D view
ax1 = fig.add_subplot(221, projection="3d")
ax1.set_facecolor("#EBF5FB")
ax1.plot_surface(GRID_A, GRID_B, GRID_PRICE,
                 alpha=PLANE_ALPHA, color=PLANE_COLOR, edgecolor="none")
ax1.scatter(df["Area_sqft"], df["Bedrooms"], df["HousePrice"],
            c=scatter_colors, s=40, edgecolors="white", linewidths=0.4, zorder=5)
ax1.set_xlabel("Area (sq ft)", fontsize=9, labelpad=8, color="#2C3E50")
ax1.set_ylabel("Bedrooms",     fontsize=9, labelpad=8, color="#2C3E50")
ax1.set_zlabel("Price (USD)",  fontsize=9, labelpad=8, color="#2C3E50")
ax1.set_title("3-D View: Regression Plane + Data", fontsize=10, fontweight="bold",
              color="#1A252F", pad=10)
ax1.view_init(elev=25, azim=-55)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar1 = fig.colorbar(sm, ax=ax1, shrink=0.5, pad=0.08)
cbar1.set_label("Actual Price (USD)", fontsize=8)
cbar1.ax.tick_params(labelsize=7)

eq_text = (
    f"Price = {b0:,.0f}\n"
    f"  + {b1:,.0f} * Area (sc)\n"
    f"  + {b2:,.0f} * Beds (sc)"
)
ax1.text2D(0.02, 0.96, eq_text, transform=ax1.transAxes,
           fontsize=7.5, va="top", color="#1A252F",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#D6EAF8",
                     edgecolor="#2980B9", alpha=0.85))


# Panel 2: top down (bird's eye view)
ax2 = fig.add_subplot(222, projection="3d")
ax2.set_facecolor("#EBF5FB")
ax2.plot_surface(GRID_A, GRID_B, GRID_PRICE,
                 alpha=PLANE_ALPHA, color=PLANE_COLOR, edgecolor="none")
ax2.scatter(df["Area_sqft"], df["Bedrooms"], df["HousePrice"],
            c=scatter_colors, s=35, edgecolors="white", linewidths=0.3, zorder=5)
ax2.set_xlabel("Area (sq ft)", fontsize=9, labelpad=8, color="#2C3E50")
ax2.set_ylabel("Bedrooms",     fontsize=9, labelpad=8, color="#2C3E50")
ax2.set_zlabel("Price (USD)",  fontsize=9, labelpad=8, color="#2C3E50")
ax2.set_title("Top-Down View (Elev=90)", fontsize=10, fontweight="bold",
              color="#1A252F", pad=10)
ax2.view_init(elev=90, azim=-90)


# Panel 3: side view
ax3 = fig.add_subplot(223, projection="3d")
ax3.set_facecolor("#EBF5FB")
ax3.plot_surface(GRID_A, GRID_B, GRID_PRICE,
                 alpha=PLANE_ALPHA, color=PLANE_COLOR, edgecolor="none")
ax3.scatter(df["Area_sqft"], df["Bedrooms"], df["HousePrice"],
            c=scatter_colors, s=35, edgecolors="white", linewidths=0.3, zorder=5)
ax3.set_xlabel("Area (sq ft)", fontsize=9, labelpad=8, color="#2C3E50")
ax3.set_ylabel("Bedrooms",     fontsize=9, labelpad=8, color="#2C3E50")
ax3.set_zlabel("Price (USD)",  fontsize=9, labelpad=8, color="#2C3E50")
ax3.set_title("Side View: Price vs Area", fontsize=10, fontweight="bold",
              color="#1A252F", pad=10)
ax3.view_init(elev=15, azim=10)


# Panel 4: residuals plot
# Blue lines = under-prediction (actual above plane)
# Red lines  = over-prediction  (actual below plane)
ax4 = fig.add_subplot(224, projection="3d")
ax4.set_facecolor("#EBF5FB")
ax4.plot_surface(GRID_A, GRID_B, GRID_PRICE,
                 alpha=0.25, color=PLANE_COLOR, edgecolor="none")
ax4.scatter(X_test[:, 0], X_test[:, 1], y_test,
            c="black", s=40, zorder=6, label="Actual price")
ax4.scatter(X_test[:, 0], X_test[:, 1], y_pred_test,
            c="steelblue", s=20, marker="x", zorder=5, label="Predicted (on plane)")

for i in range(len(X_test)):
    ax_val  = X_test[i, 0]
    bed_val = X_test[i, 1]
    y_act   = y_test[i]
    y_pr    = y_pred_test[i]
    colour  = "#2980B9" if y_act >= y_pr else "#E74C3C"
    ax4.plot3D(
        [ax_val,  ax_val],
        [bed_val, bed_val],
        [y_act,   y_pr],
        color=colour, linewidth=1.0, alpha=0.8
    )

ax4.set_xlabel("Area (sq ft)", fontsize=9, labelpad=8, color="#2C3E50")
ax4.set_ylabel("Bedrooms",     fontsize=9, labelpad=8, color="#2C3E50")
ax4.set_zlabel("Price (USD)",  fontsize=9, labelpad=8, color="#2C3E50")
ax4.set_title("Residuals in 3-D (Test Set)\nBlue=Under-prediction  Red=Over-prediction",
              fontsize=10, fontweight="bold", color="#1A252F", pad=10)
ax4.view_init(elev=25, azim=-55)
ax4.legend(fontsize=7, loc="upper left")


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("house_price_3d_regression.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\nSaved -> house_price_3d_regression.png")
plt.show()


# ------------------------------------------------------------------
# Business Questions
# ------------------------------------------------------------------

print("\n" + "=" * 60)
print("BUSINESS QUESTIONS")
print("=" * 60)

# unscale coefficients to interpret in original units
area_std = np.sqrt(scaler.var_[0])
bed_std  = np.sqrt(scaler.var_[1])
b1_orig  = b1 / area_std
b2_orig  = b2 / bed_std

print(f"""
Q1. How much does price increase per extra square foot?
   -> Each additional sq ft adds ~${b1_orig:,.2f} to the price.
   (The real coefficient I used to generate data was $120/sqft)

Q2. How much does an extra bedroom add?
   -> Each bedroom adds ~${b2_orig:,.2f} to the predicted price.
   (True coefficient was $18,000/bedroom)

Q3. Is the model a good fit?
   -> Test R2 = {r2_test:.4f} means model explains {r2_test*100:.1f}% of price variance.
   -> RMSE = ${rmse_test:,.0f} - prediction off by this much on average.
   -> That's about {rmse_test/df['HousePrice'].mean()*100:.1f}% of the mean price (${df['HousePrice'].mean():,.0f}).

Q4. Which feature has more impact?
   -> Scaled coefficients: Area = {b1:,.0f}, Bedrooms = {b2:,.0f}
   -> {'Area has more impact' if abs(b1) > abs(b2) else 'Bedrooms has more impact'} based on scaled coefficients.

Q5. What does the regression plane mean in 3D?
   -> The plane is all (Area, Bedrooms, PredictedPrice) combinations
      the model can generate. Real houses either sit above (under-predicted)
      or below (over-predicted) the plane. The vertical distance to the
      plane is the residual/error for that house.

Q6. Why are some houses far from the plane?
   -> Location, condition, age, renovations - things our model doesn't know.
   A house in a premium area will always appear above the plane because
   we didn't include location as a feature.

Q7. How to improve this model?
   -> More features: location score, year built, garage, garden size
   -> Non-linear model: Random Forest or Gradient Boosting
   -> Polynomial features like Area^2 or Area*Bedrooms interaction
   -> Transform the target: use log(Price) to reduce skewness
""")


# ------------------------------------------------------------------
# Predict some new houses
# ------------------------------------------------------------------

print("New House Predictions:")
print(f"  {'Area (sq ft)':>14} {'Bedrooms':>10} {'Predicted Price':>18}")
print(f"  {'-'*46}")

houses = [
    {"Area_sqft": 1500, "Bedrooms": 3},
    {"Area_sqft": 2800, "Bedrooms": 5},
    {"Area_sqft": 800,  "Bedrooms": 2},
    {"Area_sqft": 3500, "Bedrooms": 4},
]

for h in houses:
    X_new    = np.array([[h["Area_sqft"], h["Bedrooms"]]])
    X_new_sc = scaler.transform(X_new)
    pred     = model.predict(X_new_sc)[0]
    print(f"  {h['Area_sqft']:>14,.0f} {h['Bedrooms']:>10.0f} ${pred:>16,.2f}")

print(f"""
Model Summary:
  Model          : Multivariate Linear Regression (OLS)
  Features       : Area (sq ft), Number of Bedrooms
  Target         : House Price (USD)
  Training       : {len(X_train)} samples
  Testing        : {len(X_test)} samples
  Scaling        : StandardScaler

  Fitted equation (original units approx):
    Price = intercept + {b1_orig:,.2f} * Area_sqft + {b2_orig:,.2f} * Bedrooms
""")

print("Done!")
