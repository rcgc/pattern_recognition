import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('./iris.csv')
# print(df.head())

# Assume the class column is named 'Species'
if 'Species' not in df.columns:
    raise ValueError("Expected a 'Species' column in your CSV.")

# Set up plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Assign fixed colors to each species: red, blue, orange
color_list = ['red', 'blue', 'orange']
class_colors = dict(zip(sorted(df['Species'].unique()), color_list))

def ols_fit(X, y):
    """Compute OLS coefficients manually"""
    X_design = np.column_stack((np.ones(X.shape[0]), X))  # Add intercept term
    beta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
    return beta  # [intercept, slope]

def r_squared(y_true, y_pred):
    """Compute R^2 (coefficient of determination) manually"""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot

# Plot 1: Sepal
print("\n--- Sepal Regression Equations (Manual OLS) ---")
for species in df['Species'].unique():
    subset = df[df['Species'] == species]
    X = subset[['SepalLengthCm']].values
    y = subset['SepalWidthCm'].values
    beta = ols_fit(X, y)
    intercept, slope = beta
    y_pred = intercept + slope * X.ravel()
    r2 = r_squared(y, y_pred)

    print(f"{species}:")
    print(f"  Regression coefficient (slope): {slope:.4f}")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  Coefficient of determination (R²): {r2:.4f}")
    print(f"  Equation: SepalWidthCm = {slope:.4f} * SepalLengthCm + {intercept:.4f}")

    x_vals = np.linspace(X.min(), X.max(), 100)
    y_vals = intercept + slope * x_vals
    axes[0].scatter(X, y, color=class_colors[species], alpha=0.5, label=species)
    axes[0].plot(x_vals, y_vals, color=class_colors[species])

axes[0].set_title("SepalLengthCm vs SepalWidthCm (Manual OLS)")
axes[0].set_xlabel("SepalLengthCm")
axes[0].set_ylabel("SepalWidthCm")
axes[0].legend()

# Plot 2: Petal
print("\n--- Petal Regression Equations (Manual OLS) ---")
for species in df['Species'].unique():
    subset = df[df['Species'] == species]
    X = subset[['PetalLengthCm']].values
    y = subset['PetalWidthCm'].values
    beta = ols_fit(X, y)
    intercept, slope = beta
    y_pred = intercept + slope * X.ravel()
    r2 = r_squared(y, y_pred)

    print(f"{species}:")
    print(f"  Regression coefficient (slope): {slope:.4f}")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  Coefficient of determination (R²): {r2:.4f}")
    print(f"  Equation: PetalWidthCm = {slope:.4f} * PetalLengthCm + {intercept:.4f}")

    x_vals = np.linspace(X.min(), X.max(), 100)
    y_vals = intercept + slope * x_vals
    axes[1].scatter(X, y, color=class_colors[species], alpha=0.5, label=species)
    axes[1].plot(x_vals, y_vals, color=class_colors[species])

axes[1].set_title("PetalLengthCm vs PetalWidthCm (Manual OLS)")
axes[1].set_xlabel("PetalLengthCm")
axes[1].set_ylabel("PetalWidthCm")
axes[1].legend()

plt.tight_layout()
plt.show()
