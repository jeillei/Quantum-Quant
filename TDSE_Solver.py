import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq

def TDSE(x0=-60.0, p0=1000.0, sigma=0.1, hbar=1.0, m=1.0, N=10024, L=200.0, dt=0.1, total_steps=5000,
         V_func=None, volatility_series=None):
    
    dx = L / N
    x = np.linspace(0, L, N)
    k = 2 * np.pi * fftfreq(N, d=dx)

    def make_initial_wavefunction(x, x0, p0, sigma, hbar=1.0):
        psi0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * p0 * x / hbar)
        psi0 /= np.linalg.norm(psi0)
        return psi0

    psi = make_initial_wavefunction(x, x0, p0, sigma)
    T = np.exp(-1j * (hbar**2 * k**2 / (2 * m)) * dt / hbar)

    wavefunction_snapshots = []
    x_expect_history = []  # running memory of expectation values
    MA_window = 30         # or whatever window size you want


    for step in range(total_steps + 1):
        t_now = step * dt

        prob = np.abs(psi)**2
        prob /= np.sum(prob)
        x_expect = np.sum(x * prob)
        x_var = np.sum(prob * (x - x_expect)**2)
        sigma_t = np.sqrt(x_var)

        p_k = fft(psi)
        p = 2 * np.pi * fftfreq(len(x), d=(x[1] - x[0]))
        p_expect = np.sum(p * np.abs(p_k)**2)

        volatility = volatility_series.iloc[min(step, len(volatility_series) - 1)]

        # --- Store and compute moving average ---
        x_expect_history.append(x_expect)
        if len(x_expect_history) > MA_window:
            x_expect_history.pop(0)  # keep fixed-size buffer

        # --- Moving average for potential center ---
        if len(x_expect_history) >= MA_window:
            center = np.mean(x_expect_history)
        else:
            center = x_expect  # fallback for early steps

        # --- Pass to potential function ---
        V = V_func(x, t_now, center, x_expect, p_expect, sigma_t, volatility)

        V_half = np.exp(-1j * V * dt / (2 * hbar))

        if step % 1 == 0:  # adjust frequency if needed
            wavefunction_snapshots.append(psi.copy())

        psi *= V_half
        psi_k = fft(psi)
        psi_k *= T
        psi = ifft(psi_k)
        psi *= V_half

    return x, wavefunction_snapshots, V, []

