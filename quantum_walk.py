import numpy as np
import matplotlib.pyplot as plt
from TDSE_Solver import TDSE
import pandas as pd

seed=33
S0=100
mu=0.05
sigma_gbm=1.0
T=500

def simulate_gbm(seed, S0, mu, sigma, T):
    df = pd.read_csv("HistoricalQuotes.csv")
    df.columns = df.columns.str.strip() 
    df = df[::-1]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.iloc[0:0+T].reset_index(drop=True)
    
    for col in ["High", "Low"]:
        df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

    df["High_norm"] = df["High"] / df["High"].iloc[0] * S0
    df["Low_norm"] = df["Low"] / df["Low"].iloc[0] * S0

    if seed is not None:
        np.random.seed(seed)

    dt = 1 / T
    N = T  
    t = np.linspace(0, T, N)
    s = np.zeros(N)
    s[0] = S0

    for i in range(1, N):
        Z = np.random.normal()
        s[i] = s[i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    
    print(df.head())

    return t, s, df

def V_market_adaptive(x, t, center, x_expect, p_expect, sigma, volatility):
    revert_strength = 0.001
    trend_strength = 0.07
    V = np.zeros_like(x)

    if center is not None:
        V += revert_strength * (x - center)**2

    V -= trend_strength * (x - x_expect)

    push_center = x_expect + np.sign(p_expect) * 0
    push = np.exp(-((x - push_center)**2) / (2 * (2 * sigma)**2))
    V -= 0.1 * np.sign(p_expect) * push

    return V



t, s, Prices = simulate_gbm(seed, S0, mu, sigma_gbm, T)
from scipy.stats import linregress

# Use the first N points
N = 5
y = Prices["High_norm"].iloc[:N].values
x = np.arange(N)

# Fit: y = slope * x + intercept
slope, _, _, _, _ = linregress(x, y)

# Set initial conditions
start_price = y[0]
x0 = start_price
p0 = slope   # scale to control wave packet speed

sigma_TDSE = Prices["High_norm"].rolling(N).std().iloc[N]   # volatility -> uncertainty

x, psi_snapshots, V, _ = TDSE(
    V_func=V_market_adaptive,
    x0=x0,
    p0=p0,
    sigma= sigma_TDSE,
    L= 10000,
    total_steps=T,
    # time_dependent=True,
    volatility_series=Prices["High_norm"].rolling(10).std().fillna(method='bfill')
)

def sample_tdse_path_from_wavefunction(x, wave_snapshots, n_samples=1):
    path = []
    for psi in wave_snapshots:
        prob_density = np.abs(psi)**2
        prob_density /= np.sum(prob_density)  # normalize
        sample = np.random.choice(x, p=prob_density)
        path.append(sample)
    return np.array(path)

from filterpy.kalman import KalmanFilter

def kalman_smooth(signal, process_var=1e-4 , measurement_var=0.01):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[signal[0]]])      # initial state
    kf.F = np.array([[1]])              # state transition matrix
    kf.H = np.array([[1]])              # measurement function
    kf.P *= 1000.                       # covariance matrix
    kf.R = measurement_var              # measurement noise
    kf.Q = process_var                  # process noise

    smoothed = []
    for z in signal:
        kf.predict()
        kf.update(z)
        smoothed.append(kf.x[0, 0])
    return np.array(smoothed)



def sample_tdse_path_from_wavefunction(x, wave_snapshots, n_samples=5):
    paths = []
    for _ in range(n_samples):
        path = []
        for psi in wave_snapshots:
            prob_density = np.abs(psi)**2
            prob_density /= np.sum(prob_density)
            sample = np.random.choice(x, p=prob_density)
            path.append(sample)
        paths.append(path)
    return np.array(paths)

paths = sample_tdse_path_from_wavefunction(x, psi_snapshots, n_samples=5)

mean_path = np.mean(paths, axis=0)
std_path = np.std(paths, axis=0)

plt.figure(figsize=(12, 6))

# Plot the mean path
plt.plot(mean_path, label="Mean Quantum Path", color="blue", linewidth=2)

# Plot the ±1σ confidence band
plt.fill_between(range(len(mean_path)),
                 mean_path - 0.2 *std_path,
                 mean_path + 0.2 * std_path,
                 alpha=0.3, color="blue", label="±1σ Band")

# Optionally overlay real or GBM data
# plt.plot(t, s, label="GBM", linestyle='--', color="orange", alpha=0.7)
plt.plot(Prices["High_norm"].values[:len(mean_path)], label="High (Actual)", color="gray", alpha=0.5)
plt.plot(Prices["Low_norm"].values[:len(mean_path)], label="Low (Actual)", color="gray", alpha=0.5)

plt.title("Quantum Path Ensemble vs GBM and Actual Prices")
plt.xlabel("Time (Days)")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.plot(x, V_market_adaptive(x, t=0, x_expect=100, p_expect=5, sigma=5, volatility=0.02))
plt.title("Adaptive Market Potential")
plt.xlabel("x (Price)")
plt.ylabel("V(x)")
plt.grid(True)
plt.show()


