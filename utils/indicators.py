"""Technical Indicators"""

"""
These functions are defined to extract some indicators which will be used for training.
1. Simple Moving Average (SMA)
2. Bollinger Bands
3. Momentum
4. RSI
5. William %R
6. MACD
7. Daily return
8. Cumulative daily return

Note: We use date as index to access rows of dataframe. Date is arranged in chronological order
"""

import pandas as pd

def compute_rolling_mean(df, window=30):
    """Return rolling mean of given values, using specified window size."""
    df['SMA_{}'.format(str(window))] = df['Price'].rolling(window=window, center=False).mean()
    return df

def compute_rolling_std(df, window=30):
    """Return rolling standard deviation of given values, using specified window size."""
    df['STD_{}'.format(str(window))] = df['Price'].rolling(window=window, center=False).std()
    return df

def compute_bollinger_bands(df, window=30):
    """Return upper and lower Bollinger Bands."""
    df['SMA_{}'.format(str(window))] = compute_rolling_mean(df, window=window)
    df['SMA_{}'.format(str(window))] = compute_rolling_std(df, window=window)
    df['Upper_band_{}'.format(str(window))] = df['SMA_{}'.format(str(window))] + 2 * df['SMA_{}'.format(str(window))]
    df['Lower_band_{}'.format(str(window))] = df['SMA_{}'.format(str(window))] - 2 * df['SMA_{}'.format(str(window))]
    df['Band_value_{}'.format(str(window))] = (df['Price'] - df['SMA_{}'.format(str(window))]) / (2 * df['SMA_{}'.format(str(window))])
    return df

def compute_momentum(df, window=1):
    """Return momentum."""
    df['Momentum'] = (df['Price'] / df['Price'].shift(window) - 1)
    return df.fillna(0)

def compute_rsi(df, window=14):
    """Return RSI."""
    delta = df['Price'].diff()[1:]
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    roll_up = up.rolling(window=window).mean()
    roll_down = down.abs().rolling(window=window).mean()
    
    # Calculate the RSI based on SMA
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df['RSI'] = rsi
    return df

def compute_william_percent_r(df, window=14):
    """Return William percentage."""
    delta = df.copy()
    delta['high'] = df['Price'].rolling(window=window).max()
    delta['low'] = df['Price'].rolling(window=window).min()
    delta['williams'] = (delta['high'] - delta['Price']) / (delta['high'] - delta['low']) * -100
    df['Williams'] = delta['williams']
    return df


def compute_macd(df):
    """Return MACD."""
    delta = df.copy()
    delta['ema_12d'] = df['Price'].ewm(ignore_na=False,
                                         span=12,min_periods=0,
                                         adjust=True).mean()
    delta['ema_26d'] = df['Price'].ewm(ignore_na=False,
                                         span=26,min_periods=0,
                                         adjust=True).mean()
    delta['macd'] = delta['ema_12d'] - delta['ema_26d']
    delta['macd_signal_line'] = delta['macd'].rolling(window=9).mean()
    delta['macd_historical'] = delta['macd'] - delta['macd_signal_line']

    df['MACD'] = delta['macd_historical']
    return df

def compute_daily_return(df):
    #daily_returns = df.copy()
    #daily_returns[1:] = (df[1:]/ df[:-1].values) - 1
    #daily_returns.ix[0, :] = 0
    #df['Daily_return'] = daily_returns
    df['Daily_return'] = (df['Price']/df['Price'].shift(1)) - 1
    return df

def compute_cumulative_return(df):
    #daily_returns = df.copy()
    #daily_returns[1:] = (df[1:]/ df[:-1].values) - 1 #need to modify here
    #daily_returns.ix[0, :] = 0
    #df['Cum_daily_return'] = daily_returns
    df['Cum_daily_return'] = (df['Price']/df['Price'][0]) - 1
    return df


def fill_missing_values(df):
    """Fill missing values in data frame, in place."""
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
