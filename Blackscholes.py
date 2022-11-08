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
# example, can replace with different stock prices
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

def N(z):
#   Normal cumulative density function
#   param z: point at which cumulative density is calculated
#   return: cumulative density under normal curve
    return norm.cdf(z)

def black_scholes_call_value(S, K, r, t, vol):
#   black scholes call option
    # parameters
    # S: underlying stock price
    # K: strike price
    # r: risk free rate(annual rate expressed in terms of continuous compounding)
    # t: time to expiration
    # return black scholes call option value

    d1 = ((1.0 / (vol * np.sqrt(t))) * (np.log(S / K)) + (r + 0.5 * vol **2.0) * t)
    d2 = d1 - (vol * np.sqrt(t))

    return N(d1) * S - N(d2) * K * np.exp(-r * t)

def black_scholes_put_value(S, K, r, t, vol):
 #  black scholes call option
    # parameters
    # S: underlying stock price
    # K: strike price
    # r: risk free rate(annual rate expressed in terms of continuous compounding)
    # t: time to expiration
    # return black scholes put option value

    d1 = (1.0 / (vol * np.sqrt(t))) * (np.log(S / K) + (r + 0.5 * vol ** 2.0) * t)
    d2 = d1 - (vol * np.sqrt(t))

    return N(-d2) * K *np.exp(-r * t) - N(-d1) * S

call_value = black_scholes_call_value(S, K, r, t, vol)
put_value = black_scholes_put_value(S, K, r, t, vol)

print(f"Black-Scholes call value {call_value:.2f}")
print(f"Black-Scholes put value {put_value:.2f}")