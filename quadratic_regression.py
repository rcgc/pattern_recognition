import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('./iris.csv')

# Assume the class column is named 'Species'
if 'Species' not in df.columns:
    raise ValueError("Expected a 'Species' column in your CSV.")

# Set up plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Assign fixed colors to each species: red, blue, orange
color_list = ['red', 'blue', 'orange']
class_colors = dict(zip(sorted(df['Species'].unique()), color_list))

def ols_fit_quadratic(X, y):
    """Compute OLS coefficients manually for quadratic regression"""
    X_squared = X ** 2
    X_design = np.column_stack((np.ones(X.shape[0]), X, X_squared))  # Add intercept, x, and x^2
    beta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
    return beta  # [intercept, slope, quadratic_term]

def r_squared(y_true, y_pred):
    """Compute R^2 (coefficient of determination) manually"""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot

# Plot 1: Sepal
print("\n--- Sepal Quadratic Regression Equations (Manual OLS) ---")
for species in df['Species'].unique():
    subset = df[df['Species'] == species]
    X = subset[['SepalLengthCm']].values
    y = subset['SepalWidthCm'].values
    beta = ols_fit_quadratic(X, y)
    intercept, slope, quad = beta
    y_pred = intercept + slope * X.ravel() + quad * (X.ravel() ** 2)
    r2 = r_squared(y, y_pred)

    print(f"{species}:")
    print(f"  Coefficients: intercept = {intercept:.4f}, linear = {slope:.4f}, quadratic = {quad:.4f}")
    print(f"  Coefficient of determination (R²): {r2:.4f}")
    print(f"  Equation: SepalWidthCm = {quad:.4f} * SepalLengthCm² + {slope:.4f} * SepalLengthCm + {intercept:.4f}")

    x_vals = np.linspace(X.min(), X.max(), 100)
    y_vals = intercept + slope * x_vals + quad * (x_vals ** 2)
    axes[0].scatter(X, y, color=class_colors[species], alpha=0.5, label=species)
    axes[0].plot(x_vals, y_vals, color=class_colors[species])

axes[0].set_title("SepalLengthCm vs SepalWidthCm (Quadratic OLS)")
axes[0].set_xlabel("SepalLengthCm")
axes[0].set_ylabel("SepalWidthCm")
axes[0].legend()

# Plot 2: Petal
print("\n--- Petal Quadratic Regression Equations (Manual OLS) ---")
for species in df['Species'].unique():
    subset = df[df['Species'] == species]
    X = subset[['PetalLengthCm']].values
    y = subset['PetalWidthCm'].values
    beta = ols_fit_quadratic(X, y)
    intercept, slope, quad = beta
    y_pred = intercept + slope * X.ravel() + quad * (X.ravel() ** 2)
    r2 = r_squared(y, y_pred)

    print(f"{species}:")
    print(f"  Coefficients: intercept = {intercept:.4f}, linear = {slope:.4f}, quadratic = {quad:.4f}")
    print(f"  Coefficient of determination (R²): {r2:.4f}")
    print(f"  Equation: PetalWidthCm = {quad:.4f} * PetalLengthCm² + {slope:.4f} * PetalLengthCm + {intercept:.4f}")

    x_vals = np.linspace(X.min(), X.max(), 100)
    y_vals = intercept + slope * x_vals + quad * (x_vals ** 2)
    axes[1].scatter(X, y, color=class_colors[species], alpha=0.5, label=species)
    axes[1].plot(x_vals, y_vals, color=class_colors[species])

axes[1].set_title("PetalLengthCm vs PetalWidthCm (Quadratic OLS)")
axes[1].set_xlabel("PetalLengthCm")
axes[1].set_ylabel("PetalWidthCm")
axes[1].legend()

plt.tight_layout()
plt.show()
