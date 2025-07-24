# 📈 Quantum Finance: Schrödinger Equation for Stock Price Modeling

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project explores an unconventional use of the **time-dependent Schrödinger Equation (TDSE)** to simulate stock price behavior — comparing it against the widely-used **Geometric Brownian Motion (GBM)** model.

> ✨ *Can quantum physics capture market behavior better than stochastic finance?*

---

## 🔍 Overview

We model price evolution using:
- ⚙️ **Geometric Brownian Motion (GBM)** — the standard model for asset prices.
- 🧠 **Quantum Wavefunctions** — probability fields evolving under a dynamic market potential.

These are tested against historical stock data from **Apple Inc. (AAPL)**.

---

## 📁 Project Structure
├── HistoricalQuotes.csv # Historical data (e.g., AAPL)
├── benchmark_ar1_vs_gbm.py # Classical benchmark: AR(1) vs GBM
├── TDSE_Solver.py # Core TDSE engine (Fourier-based)
├── quantum_stock_simulation.py # Quantum market dynamics + visualization
└── README.md # This file


---

## 📈 Simulation Approaches

<details>
<summary><strong>1. Benchmark (AR(1) vs GBM)</strong></summary>

- Normalizes high/low prices to simulate stock movement
- Extracts drift and volatility from historical log returns
- Simulates:
  - 🔁 **AR(1)** process (autocorrelated returns)
  - 💹 **GBM** (stochastic differential equation)

</details>

<details>
<summary><strong>2. TDSE Quantum Market Model</strong></summary>

- Initializes wavefunction with:
  - `x₀`: starting price  
  - `p₀`: estimated trend  
  - `σ`: rolling volatility → uncertainty

- Evolves wavefunction using the **Split-Operator Fourier Method**
- Dynamically updates potential based on:
  - Recent price trends (momentum)
  - Mean-reversion
  - Market volatility
- Samples ensemble paths from the probability distribution

</details>

---

## 🔬 Sample Result

<p align="center">
  <img src="preview_chart.png" alt="Quantum vs GBM vs Actual Prices" width="600"/>
</p>

---

## ⚙️ Setup

Install dependencies with:

```bash
pip install numpy pandas matplotlib scipy yfinance filterpy
