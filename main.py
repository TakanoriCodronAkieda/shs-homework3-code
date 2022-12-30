import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def european_put(strike):
    """Returns payoff function for a European put option with given strike price."""
    def payoff(S_T):
        """Payoff at expiration."""
        return max(strike - S_T, 0)
    return payoff

def european_call(strike):
    """Returns payoff function for a European call option with given strike price."""
    def payoff(S_T):
        """Payoff at expiration."""
        return max(S_T - strike, 0)
    return payoff

def plum_payoff(S):
  if S >= 0 and S <= 10:
    payoff = 10
  elif S >= 50 and S <= 60:
    payoff = 10
  elif S >= 10 and S <= 20:
    payoff = -5*(S-10)+10
  elif S >= 20 and S <= 30:
    payoff = 8*(S-20)-40
  elif S >= 30 and S <= 40:
    payoff = -8*(S-30)+40
  elif S >= 40 and S <= 50:
    payoff = 5*(S-40)-40
  else:
    payoff = 0
  return payoff

# plot plum payoff
def plot_plum():
    # Generate a range of possible prices for the underlying asset
    S = np.linspace(0, 60, 1000)
    
    # Calculate the payoff for each price by applying the payoff function to each price
    h = [plum_payoff(S_T) for S_T in S]
    
    # Plot the payoff function
    plt.plot(S, h)
    plt.xlabel("Price of the underlying asset")
    plt.ylabel("Custom payoff function")
    plt.show()

def plot_european_put(strike):
    # Generate a range of possible prices for the underlying asset
    S = np.linspace(0, 2*strike, 1000)
    
    # Calculate the payoff for each price by applying the payoff function to each price
    h = [european_put(strike)(S_T) for S_T in S]
    
    # Plot the payoff function
    plt.plot(S, h)
    plt.xlabel("Price of the underlying asset")
    plt.ylabel("Payoff of the European derivative")
    plt.title(f"European derivative with strike price {strike}")
    plt.show()

def binomial_model(r, S_0, N, Delta, U, D, h):
    dt = Delta / N

    # Create DataFrames to store prices and values of the derivative
    S = pd.DataFrame(index=range(N+1), columns=range(N+1))  # Prices of the underlying asset
    P = pd.DataFrame(index=range(N+1), columns=range(N+1))  # Payoffs of the derivative
    B = pd.DataFrame(index=range(N+1), columns=range(N+1))  # Replicating portfolio b coefficients
    A = pd.DataFrame(index=range(N+1), columns=range(N+1))  # Replicating portfolio a coefficients
    
    # Generate binomial tree of prices for the underlying asset
    S.iloc[0, 0] = S_0  # Initial price of the asset
    for i in range(N):  # Loop over periods
        for j in range(i+1):  # Loop over possible up and down moves
            S.iloc[i+1, j] = S.iloc[i, j] * D  # Calculate down move
            S.iloc[i+1, j+1] = S.iloc[i, j] * U  # Calculate up move
    
    # Compute last period payoffs
    for j in range(N+1):
        P.iloc[N, j] = h(S.iloc[N, j])
    # Populate the rest of the DataFrame with backward induction
    for i in range(N-1, -1, -1):  # Loop over periods
        for j in range(i+1):  # Loop over possible up and down moves
            P.iloc[i, j] = np.exp(-r * dt) * (P.iloc[i+1, j] + P.iloc[i+1, j+1]) / 2 # continous compounding
            # P.iloc[i, j] = (P.iloc[i+1, j] + P.iloc[i+1, j+1]) / (2 * (1 + dt * r)) # discrete compounding

    # Compute replicating portfolio B
    for i in range(N-1, -1, -1):  # Loop over periods
        for j in range(i+1):  # Loop over possible up and down moves
            B.iloc[i, j] = (P.iloc[i+1, j+1] - P.iloc[i+1, j]) / (S.iloc[i+1, j+1] - S.iloc[i+1, j])

    # Compute replicating portfolio A
    A = P - B * S

    return S, P, A, B

# Test cases for the binomial_model function
tests = [
    # Test case 1
    {
        "input": {
            "r": 0.05,
            "S_0": 8,
            "N": 3,
            "Delta": 5,
            "U": 2,
            "D": 0.5,
            "h": european_put(strike=40),
        },
        "description": "Example from lecture 7 part 2 (see https://tube.switch.ch/videos/HmXN79Stwu at 14 min 27 sec)"
    },
    # Test case 2
    {
        "input": {
            "r": 0.0125,
            "S_0": 30,
            "N": 30,
            "Delta": 30/365,
            "U": np.exp(0.18 * np.sqrt(365)), 
            "D": 1 / np.exp(0.18 * np.sqrt(365)),
            "h": plum_payoff,
        },
        "description": "Homework 3 - question 3: Plum"
    },
]

# Evaluate test cases
for i, test in enumerate(tests):
    if d := test.get("description"):
        print(f"\nTest case {i+1} - {d}:")
    else:
        print(f"\nTest case {i+1}:")

    x_in = test["input"] # these are the parameters of the test case
    
    # Print result of the function
    S, P, A, B = binomial_model(**x_in)

    initial_derivative_price = P.iloc[0, 0]
    initial_replicating_a = A.iloc[0, 0]
    initial_replicating_b = B.iloc[0, 0]

    print(f"Initial derivative price: {initial_derivative_price}")
    print(f"Initial replicating portfolio a: {initial_replicating_a}")
    print(f"Initial replicating portfolio b: {initial_replicating_b}")