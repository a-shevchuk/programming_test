import numpy as np
import pandas as pd

# Added packages
import math
#import matplotlib
#import matplotlib.pyplot as plt

# Logic


def check_uniqueness(lst):
    """
    Check if a list contains only unique values.
    Returns True only if all values in the list are unique, False otherwise
    """

    if(len(set(lst)) == len(lst)):
        return True
    else:
        return False

# Testing
data = pd.read_csv('/Users/annashevchuk/Desktop/programming_test/data/data.csv') # importing the data
lst = data['SX5T Index'].values.tolist() # converting the column 'SX5T Index' to a list
out_1 = check_uniqueness(lst) # application of the function
print(out_1)


def smallest_difference(array):
    """
    Code a function that takes an array and returns the smallest
    absolute difference between two elements of this array
    Please note that the array can be large and that the more
    computationally efficient the better
    """
    #pass

    arr = sorted(array) # sorting the array
 
    diff = 10**20 # initializing the difference as infinite
 
    for i in range(len(arr)-1): # finding the min difference by comparing pairs of values
        if abs(arr[i+1] - arr[i]) < diff:
            diff = abs(arr[i+1] - arr[i])
 
    return diff

# Testing
array = data['SX5T Index'].to_numpy() # converting the column 'SX5T Index' to an array
out_2 = smallest_difference(array) # application of the function
print(out_2) # the result if coherent – as the previous function returns False, the smallest difference should be  equal to 0.0


# Finance and DataFrame manipulation

def macd(prices, window_short=12, window_long=26):
    """
    Code a function that takes a DataFrame named prices and
    returns it's MACD (Moving Average Convergence Difference) as
    a DataFrame with same shape
    Assume simple moving average rather than exponential moving average
    The expected output is in the output.csv file
    """
    # Computation of the simple moving averages
    prices['MA_window_short'] = prices['SX5T Index'].rolling(window_short).mean() # intermediate steps, these columns can be removed
    prices['MA_window_long'] = prices['SX5T Index'].rolling(window_long).mean()

    # Computation of the MACD
    prices['macd_12_26'] =  prices['MA_window_short'] - prices['MA_window_long']

    # Replacing the NaN values by empty cells 
    prices = prices.fillna('')

    return prices

# Testing
print(macd(data))

def sortino_ratio(prices):
    """
    Code a function that takes a DataFrame named prices and
    returns the Sortino ratio for each column
    Assume risk-free rate = 0
    On the given test set, it should yield 0.05457
    """
    # Creation of a new DataFrame
    prices_returns = prices

    # Calculation of the daily returns
    prices_returns['Returns'] = prices_returns['SX5T Index'].pct_change(1)

    # Annualized return
    # (1) Daily return over the entire period
    total_return =  (prices_returns['SX5T Index'].iloc[-1] - prices_returns['SX5T Index'].iloc[0])/(prices_returns['SX5T Index'].iloc[0])
    # (2) Annualized return
    annualized_return = pow(1 + total_return,252/len(prices_returns['SX5T Index']))-1

    #Annualized volatility (such as the daily return is <0 only)
    annualized_volatility_neg = np.std(prices_returns['Returns'][prices_returns['Returns']<0]) * math.sqrt(252)

    # Computation of the Sortino ratio
    return round(annualized_return / annualized_volatility_neg,5)

# Testing
print('Sortino ratio:')
print(sortino_ratio(data)) # a little bit different from the expected result...


def expected_shortfall(prices, level=0.95):
    """
    Code a function that takes a DataFrame named prices and
    returns the expected shortfall at a given level
    On the given test set, it should yield -0.03468
    """
    # Creation of a new DataFrame
    prices_returns = prices

    # Calculation of the daily returns
    returns = prices_returns['SX5T Index'].pct_change(1)

    # Sorting of the returns
    returns.sort_values(inplace=True, ascending=True)

    # VaR 95%
    VaR_95 = returns.quantile(0.05).round(4)

    # CVaR 95%
    CVaR_95 = returns[returns <= VaR_95].mean()

    return round(CVaR_95,5)

# Testing
print('Expected Shortfall:')
print(expected_shortfall(data)) # equal to the expected result


# Plot

def visualize(prices, path):
    """
    Code a function that takes a DataFrame named prices and
    saves the plot to the given path
    """
    fig = plt.figure()
    prices.plot(x='date', y='SX5T Index', kind='scatter')
    plt.show()

    fig.savefig(path + '/graph.png')

# Testing
visualize(data,'/Users/annashevchuk/Desktop/programming_test')
# Error VS Code : "No module named 'matplotlib'". I tried the pip install command, after
# what I reopend VS Code but it still doesn't work, so I could not test this function.
# I am sorry! I could not fix this issue
