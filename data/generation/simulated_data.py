import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t, rankdata
from scipy.linalg import cholesky
import os

# === Simulation Parameters ===
T = 1000           # Number of time steps
base_corr = 0.4    # Base off-diagonal correlation
np.random.seed(42) # For reproducibility

# === Resolve Paths ===
# Get current directory where simulated_data.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to real data
real_data_path = os.path.join(current_dir, "../Real Data/real_asset_log_returns_extreme.csv")

# Directory to save simulation results
save_dir = os.path.join(current_dir, "../Simulation")
os.makedirs(save_dir, exist_ok=True)


def inverse_empirical_cdf(u_values, real_samples):
    """Map uniform values to real returns using empirical inverse CDF"""
    sorted_samples = np.sort(real_samples)
    n = len(sorted_samples)
    idx = (u_values * n).astype(int).clip(0, n - 1)
    return sorted_samples[idx]


def generate_simulation(nu=5, shock=False, volatility_cluster=False, filename="simulated_copula.csv"):
    """Generate simulated market data"""
    # Load real data
    real_df = pd.read_csv(real_data_path)

    # Drop Date column if it exists
    if "Date" in real_df.columns:
        real_df = real_df.drop(columns=["Date"])

    # Automatically set N to match number of columns
    N = real_df.shape[1]
    print(f"📥 Loaded real asset returns shape: {real_df.shape}")

    # Step 1: Generate correlated t-Copula samples
    Sigma = np.full((N, N), base_corr)
    np.fill_diagonal(Sigma, 1.0)
    L = cholesky(Sigma, lower=True)

    Z = np.random.standard_t(df=nu, size=(T, N))  # t-distributed uncorrelated samples
    X_tilde = Z @ L.T                              # Induce correlation

    # Extreme event: Add system-wide shock
    if shock:
        print("⚠️ Adding system-wide extreme shocks...")
        shock_indices = np.random.choice(T, size=int(0.05 * T), replace=False)
        shock_magnitude = np.random.uniform(-6, -4, size=(len(shock_indices), N))
        X_tilde[shock_indices] += shock_magnitude

    # Volatility clustering: Amplify variance
    if volatility_cluster:
        print("⚠️ Adding volatility clustering...")
        volatility = np.random.gamma(shape=2.0, scale=1.0, size=T)
        X_tilde *= volatility[:, np.newaxis]

    # Transform to uniform
    U_tilde = t.cdf(X_tilde, df=nu)

    # Map uniform values to real returns
    Y_tilde = np.zeros_like(U_tilde)
    for i in range(N):
        Y_tilde[:, i] = inverse_empirical_cdf(U_tilde[:, i], real_df.iloc[:, i].values)

    # Save to CSV
    save_path = os.path.join(save_dir, filename)
    pd.DataFrame(Y_tilde, columns=[f"asset_{i+1}" for i in range(N)]).to_csv(save_path, index=False)
    print(f"✅ Saved simulated returns to: {save_path}")

    # Plot distribution for Asset 1
    plt.figure(figsize=(8, 4))
    sns.histplot(Y_tilde[:, 0], bins=20, kde=True)
    plt.title(f"{filename}: Asset 1 Return Distribution")
    plt.xlabel("Return")
    plt.tight_layout()
    plt.show()


# === Generate Multiple Market Scenarios ===
print("\n=== Generating Market Scenarios ===")
generate_simulation(nu=5, filename="simulated_copula_returns.csv")               # Standard
generate_simulation(nu=2, filename="simulated_copula_fat_tail.csv")              # Fat Tails
generate_simulation(nu=5, shock=True, filename="simulated_copula_extreme.csv")   # Systemic Shocks
generate_simulation(nu=5, volatility_cluster=True, filename="simulated_copula_vol_cluster.csv")  # Volatility Clustering

