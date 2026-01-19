import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Data
# ---------------------------
y = np.array([4.0, 5.8, 8.2, 10.0])
x = np.array([2.1, 3.0, 4.1, 5.2])

# Design matrix with intercept
X = np.column_stack([np.ones_like(x), x])

# ---------------------------
# OLS Estimation
# ---------------------------
XtX_inv = np.linalg.inv(X.T @ X)
beta_hat = XtX_inv @ (X.T @ y)

# Fitted values and residuals
y_hat = X @ beta_hat
residuals = y - y_hat

# Estimate sigma^2 (your requested formula: 1/n * sum residual^2)
sigma2_hat = np.mean(residuals**2)


# ---------------------------
# Prediction standard errors
# Each prediction x_i has:
#   Var(y_hat_i) = x_i^T (XtX_inv) x_i * sigma^2
# ---------------------------
pred_var = np.array([xrow @ XtX_inv @ xrow * sigma2_hat for xrow in X])
pred_se = np.sqrt(pred_var)


# ---------------------------
# Plot with error bars
# ---------------------------
plt.figure(figsize=(7,5))
plt.scatter(x, y, label="Observed", s=70)
plt.plot(x, y_hat, label="Fitted Line")

# Error bars using predicted SEs
plt.errorbar(
    x,
    y_hat,
    yerr=pred_se,
    fmt="o",
    capsize=5,
    label="Prediction ± SE",
)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("OLS Fit with Delta-Method Prediction SEs")
plt.show()

# Print results for inspection
print("beta_hat =", beta_hat)
print("sigma^2_hat =", sigma2_hat)
print("prediction SEs =", pred_se)