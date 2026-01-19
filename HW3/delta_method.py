import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Load the data
data = np.load("global_co2_emissions.npz")
co2 = np.log10(data['co2'])  # take log of CO2
years = data['year']

# limit to last 50 years! (forgot to do this at first)
co2 = co2[-50:]
years = years[-50:]

# Set up X matrix (intercept + year)
X = np.zeros((years.shape[0], 2))
X[:,0] = 1  # intercept
X[:,1] = years.flatten()  # year
print(X)
y = co2


# Do linear regression

Beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
print('this is beta', Beta)
y_hat = X @ Beta
diff = y_hat - y  # residuals
print(diff.shape)
print('here is diff')

# Plot data and regression line
plt.scatter(X[:,1], y, label="Observed Data")
plt.plot(X[:,1], y_hat, label="Linear Model", color="red", linewidth=2, linestyle="--")
plt.title('Log CO2 emissions by year (linear model)')
plt.xlabel('Year')
plt.ylabel('CO2 emissions (log)')
plt.legend()
plt.show()

# Predict when CO2 will exceed 60,000
g_t = (np.log10(60000) - Beta[0]) / Beta[1]
print("Predicted year CO2 exceeds 60,000:", g_t)

# Get covariance matrix for Beta
n, p = X.shape
sigma_hat_squared = np.sum(diff**2) / (50) # unbiased estimate of variance

print(n - p)
print('this is n-p®')
cov_Beta = sigma_hat_squared * np.linalg.inv(X.T @ X)
print("Predicted CO2 exceeds 60,000:", cov_Beta)

# Eigenvalues of covariance matrix
eigenvalues = np.linalg.eigvals(cov_Beta)
print("Eigenvalues of Cov(Beta):", eigenvalues)
print("Square roots of eigenvalues:", np.sqrt(eigenvalues))
print('condition number')
print(np.linalg.cond(cov_Beta)) # didnt realize there was a python func for this!


# Delta method: get std error for g_t
grad = np.array([-1 / Beta[1] , -(np.log10(60000) - Beta[0]) / (Beta[1]**2)])
print('this is grad')
print(grad)
var_g = grad.T @ cov_Beta @ grad
st_error = np.sqrt(var_g)
print("Std error of predicted year (from delta method):", st_error)

# 95% confidence interval
ci_lower = g_t - 1.96 * st_error
ci_upper = g_t + 1.96 * st_error
print("95% CI for predicted year:", ci_lower, "-", ci_upper)

# Check residuals
plt.hist(diff, bins=15, edgecolor='k')
plt.ylabel(r'Residual Frequency $(\hat{y} - y)$')
plt.title("Histogram of Residuals")
plt.show()

# Plot residuals over time to check IID assumption
plt.scatter(years, diff, color='purple')
plt.axhline(0, color='black', linestyle='--')  # reference line at 0
plt.xlabel("Year")
plt.ylabel(r'Residual $(\hat{y} - y)$')
plt.title("Residuals vs Year")
plt.show()