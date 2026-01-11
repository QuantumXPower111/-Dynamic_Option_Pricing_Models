import yfinance as yf
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm

def compare_approaches(ticker='SPY'):  # Updated with real data
    data = yf.download(ticker, period='1y')
    returns = data['Close'].pct_change().dropna()
    sigma = np.std(returns) * np.sqrt(252)  # Annualized volatility from 2026 data
    S0 = data['Close'][-1]  # Current price
    K = S0 * 1.05
    T = 1
    r = 0.05  # Assume; could fetch from API
    print(f"Using real {ticker} data: S0=${S0:.2f}, sigma={sigma:.2f}")

    # Rest of original code...
    # (Truncated for brevity; include full comparison as in original)
compare_approaches()