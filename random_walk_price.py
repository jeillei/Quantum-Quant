import numpy as np
import matplotlib.pyplot as plt
import yfinance
import pandas as pd

df = pd.read_csv("HistoricalQuotes.csv")

symbol = "AAPL"
days = 500
mu = 0.1
sigma = 0.25
dt = 1/days
start_price = 150

df["Date"] = pd.to_datetime(df["Date"])
df = df.head(days)

for col in [" High", " Low"]:
    df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

df["High_norm"] = df[" High"] / df[" High"].iloc[0] * start_price
df["Low_norm"] = df[" Low"] / df[" Low"].iloc[0] * start_price

df["log_returns"] = np.log((df[" High"]/ df[" High"].shift(1)).dropna())
r_mean = df['log_returns'].mean()
r_std = df['log_returns'].std()

sig_annual = r_std * (days)**0.5
mu_annual = r_mean * (days) + 1/2 * sig_annual**2

r_t = df['log_returns'].iloc[1:]
r_t_m = df['log_returns'].shift(1).iloc[1:]

phi = (r_t * r_t_m).sum() / (r_t_m**2).sum()
print(phi)

np.random.seed(0)
s = [start_price]
s_o = [start_price]
r = [np.random.normal(0, r_std)]
for _ in range(1, days):
    eps = np.random.normal(0, r_std)
    r_t = phi * r[-1] + eps
    r.append(r_t)
    s.append(s[-1] * np.exp(r_t))
    
Z = np.random.normal(size = days)
for z in Z:
     s_o.append(s_o[-1] * np.exp((mu_annual - 1/2 * sig_annual**2)*dt + sig_annual * (dt)**0.5 * z))

s = np.array(s)
s_o = np.array(s_o)

plt.figure(figsize=(12,8))
plt.plot(s, label= "Simulated AR1")
plt.plot(s_o, label="Simulated GBM")
plt.plot(np.arange(len(df)), df["High_norm"].to_numpy())
plt.plot(np.arange(len(df)), df["Low_norm"].to_numpy())
plt.legend()
plt.show()
