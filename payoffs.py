import time
import datetime as dt
from math import sqrt, pi

import numpy as np
import pandas as pd
import yfinance as yf
import scipy as scipy

import matplotlib.pyplot as plt
import matplotlib as mat

from pandas_datareader.yahoo.options import Options
from pandas_datareader.yahoo.daily import YahooDailyReader
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import interp1d

mat.style.use("ggplot")

# underlying stock price
S = 45.0
# series of underlying stock prices to demonstrate a payoff profile
S_ = np.arange(35.0, 55.0, 0.01)
# strike price
K = 45.0
# time to expiration (you'll see this as T-t in the equation)
t = 164.0 / 365.0
# risk free rate (there's nuance to this which we'll describe later)
r = 0.02
# volatility (latent variable which is the topic of this talk)
vol = 0.25
# black scholes prices for demonstrating trades
atm_call_premium = 3.20
atm_put_premium = 2.79
otm_call_premium = 1.39
otm_put_premium = 0.92

# use a lambda for a payoff functions
# equivelant to:
#
# def call_payoff(S, K):
# return np.maximum(S - K, 0.0)
call_payoff = lambda S, K: np.maximum(S_ - K, 0.0)
# equivelant to:
#
# def put_payoff(S, K):
# return np.maximum(K - S, 0.0)
put_payoff = lambda S, K: np.maximum(K - S_, 0.0)

# plot the call payoff
plt.figure(1, figsize=(7, 4))
plt.title("Call option payoff at expiration")
plt.xlabel("Underlying stock price, S")
plt.axhline(y=0, lw=1, c="grey")
plt.plot(S_, -atm_call_premium + call_payoff(S_, K))

# plot the put payoff
plt.figure(2, figsize=(7, 4))
plt.title("Put option payoff at expiration")
plt.xlabel("Underlying stock price, S")
plt.axhline(y=0, lw=1, c="grey")
plt.plot(S_, -atm_put_premium + put_payoff(S_, K))

# plot a long straddle payoff
long_straddle = call_payoff(S_, K) + put_payoff(S_, K)
long_straddle_premium = -atm_call_premium - atm_put_premium
plt.figure(3, figsize=(7, 4))
plt.title("Long straddle payoff at expiration")
plt.xlabel("Underlying stock price, S")
plt.axhline(y=0, lw=1, c="grey")
plt.plot(S_, long_straddle_premium + long_straddle)

# plot a short straddle payoff
short_straddle = -call_payoff(S_, K) - put_payoff(S_, K)
short_straddle_premium = atm_call_premium + atm_put_premium
plt.figure(4, figsize=(7, 4))
plt.title("Short straddle payoff at expiration")
plt.xlabel("Underlying stock price, S")
plt.axhline(y=0, lw=1, c="grey")
plt.plot(S_, short_straddle_premium - long_straddle)

# plot a short iron condor payoff
short_iron_condor = (
call_payoff(S_, K + 5)
- call_payoff(S_, K)
- put_payoff(S_, K)
+ put_payoff(S_, K - 5)
)
short_iron_condor_premium = (
-otm_call_premium + atm_call_premium + atm_put_premium - otm_put_premium
)
plt.figure(5, figsize=(7, 4))
plt.title("Short iron condor payoff at expiration")
plt.xlabel("Underlying stock price, S")
plt.axhline(y=0, lw=1, c="grey")
plt.plot(S_, short_iron_condor_premium + short_iron_condor)