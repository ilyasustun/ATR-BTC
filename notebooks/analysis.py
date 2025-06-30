# %% [markdown]
# # Trade Analysis of BTC
# Ilyas Ustun  
# June 15, 2025  
# Chicago, IL   
# 
# 
# Table of Contents
# ----------------
# 1. Data Exploration and Preprocessing
# 2. Technical Indicators
# 3. Find Long and Short Trades
# 4. Backtesting
# 5. Results
#     - Trade Statistics
#     - Visualizations of Trade Statistics
#     - Long vs Short Trades Statistics
#     - Equity Curve Plot
#     - Analysis of Seasonality Patterns
# 6. Optimization: Finding Better Set of Parameters

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# %%
# Read data from the data folder
# Assuming CSV file - adjust file path and read function as needed
file_path = 'data/BTC_USDT Perpetual Bybit V5, Time - Time - 15m, 1_1_2024 120000 AM-12_31_2024 120000 AM_3c007f81-be0d-41a8-a36c-1ed6d0b7646f.csv'  # Replace with your actual file name
df = pd.read_csv(file_path, sep=";")

# Display the first few rows of the data to verify it was loaded correctly
df.head()

# %% [markdown]
# # Data Exploration and Preprocessing
# 
# This notebook analyzes Bitcoin (BTC/USDT) perpetual futures data from Bybit for the entire year 2024, using 15-minute intervals.
# 
# ## Dataset Overview
# - **Time Period:** January 1, 2024 to December 31, 2024
# - **Interval:** 15-minute candlesticks
# - **Data Points:** 35,059 entries
# - **Key Features:** Open, High, Low, Close prices, Volume, and derived technical indicators
# 
# ## Technical Indicators
# - **EMA (10):** 10-period Exponential Moving Average of closing prices
# - **ATR (10):** 10-period Average True Range, measuring market volatility
# 
# The dataset provides a comprehensive view of Bitcoin price movements throughout the year, allowing for detailed technical analysis and pattern identification.

# %% [markdown]
# ## Data Cleaning and Initial Preprocessing
# 
# Before proceeding with analysis, I'll clean the dataset by:
# 
# 1. **Removing unused columns**: The "Unnamed: 11" column contains only missing values and will be dropped.
# 2. **Handling time-related columns**: The dataset includes both "Time left" (now the index) and "Time right" columns. The "Time right" column contains only the time portion of the timestamp and is redundant since we're using "Time left" as our datetime index.
# 
# The cleaned dataset will include:
# - **Price data**: Open, High, Low, Close, Median, Typical, Weighted
# - **Volume data**: Volume and Quote asset volume
# - **Technical indicators**: EMA_10 (10-period Exponential Moving Average) and ATR_10 (10-period Average True Range)
# 
# This cleaning ensures we work with only relevant data for our analysis and visualization in subsequent steps.

# %%
# Duplicate rows in the data
df.loc[df.duplicated(subset=['Time left'], keep=False)]

# %%
# Drop unused and redundant columns
df = df.drop(columns=['Unnamed: 11', 'Time right'])

# Verify the columns were dropped
print(f"Remaining columns: {df.columns.tolist()}")
print(f"Dataset shape: {df.shape}")

# Drop any duplicate rows based on "Time left"
df.drop_duplicates(subset=['Time left'], keep='first', inplace=True)
print(f"Dataset shape after removing duplicates: {df.shape}")

# Check if there are still any duplicate rows in the data
df.loc[df.duplicated(subset=['Time left'], keep=False)]

# %%
# Convert time columns to datetime format
df['Time left'] = pd.to_datetime(df['Time left'])
df.set_index('Time left', inplace=True)

# Ensure df is sorted by date in ascending order  
# This is typically done before passing data to the function, but we can check it here
if not df.index.is_monotonic_increasing:
    raise ValueError("Data must be sorted by date in ascending order.")
else:
    print("Data is monotonically increasing")

# Display the first few rows of the cleaned dataset
df.head(3)

# %% [markdown]
# ### Data Preprocessing and Technical Indicators
# 
# To make our technical indicator calculations more flexible, I've created modular functions for calculating:
# 
# 1. **Exponential Moving Average (EMA)** with adjustable periods: A type of moving average that gives more weight to recent price data, making it more responsive to new information.
# 2. **Average True Range (ATR)** with adjustable periods: A volatility indicator that measures market volatility by decomposing the entire range of an asset price for a period.
# 
# 

# %%
def calculate_ema(data, column='Close', period=10, adjust=False):
    """
    Calculate Exponential Moving Average for any period
    
    Parameters:
    - data: pandas DataFrame containing price data
    - column: column name to calculate EMA on (default: 'Close')
    - period: number of periods for EMA calculation (default: 10)
    - adjust: whether to adjust for bias in the beginning (default: False)
    
    Returns:
    - pandas Series containing EMA values
    """

    ema = data[column].ewm(span=period, adjust=adjust).mean()
    return ema


def calculate_tr(data):
    """
    Calculate True Range
    
    Parameters:
    - data: pandas DataFrame containing OHLC price data
    
    Requirements:
    - data must contain 'High', 'Low', and 'Close' columns
    - 'High', 'Low', 'Close' must be numeric types
    - data must be a pandas DataFrame
    - data must be sorted by date in ascending order
    - data must not be empty
    - data must not contain duplicate dates
    - data must not contain any non-numeric values in 'High', 'Low', 'Close' columns

    Returns:
    - pandas Series containing ATR values
    """
    # Calculate True Range
    tr = np.maximum(
        data['High'] - data['Low'],
        abs(data['High'] - data['Close'].shift(1)),
        abs(data['Low'] - data['Close'].shift(1))
        )
    
    # Return TR as a pandas Series
    return tr

def calculate_atr(data, period=10):
    """
    Calculate Average True Range for any period
    
    Parameters:
    - data: pandas DataFrame containing OHLC price data
    - period: number of periods for ATR calculation (default: 10)
    
    Requirements:
    - data must contain 'High', 'Low', and 'Close' columns
    - 'High', 'Low', 'Close' must be numeric types
    - data must be a pandas DataFrame
    - period must be a positive integer
    - data must be sorted by date in ascending order
    - data must have at least 'period' number of rows
    - data must not be empty
    - data must not contain duplicate dates
    - data must not contain any non-numeric values in 'High', 'Low', 'Close' columns

    
    Returns:
    - pandas Series containing ATR values
    """

    # Calculate True Range if not already present
    if 'TR' not in data.columns:
        data['TR'] = calculate_tr(data)
    
    # Ensure 'TR' column is numeric
    if not pd.api.types.is_numeric_dtype(data['TR']):
        raise ValueError("The 'TR' column must contain numeric values.")    
    
    # Ensure data has enough rows for the ATR calculation
    if len(data) < period:
        raise ValueError(f"Data must have at least {period} rows for ATR calculation.")
    
    # Ensure period is a positive integer
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer.")

        
    # Calculate ATR as a pandas Series
    atr = data['TR'].rolling(window=period).mean()
    return atr

# %%
def add_multiple_indicators(df, ema_periods=[20, 50, 100, 200], atr_periods=[10, 20, 50]):
    """
    Add multiple EMA and ATR indicators to the dataframe with different periods
    
    Parameters:
    - df: pandas DataFrame containing price data
    - ema_periods: list of periods for EMA calculation
    - atr_periods: list of periods for ATR calculation
    
    Returns:
    - DataFrame with additional indicators
    """
    # Add EMAs with different periods
    for period in ema_periods:
        column_name = f'EMA_{period}'
        if column_name not in df.columns:
            df[column_name] = calculate_ema(df, column='Close', period=period)
    
    # Add ATRs with different periods (reusing the existing TR column)
    for period in atr_periods:
        column_name = f'ATR_{period}'
        if column_name not in df.columns:
            df[column_name] = calculate_atr(df, period=period)
    
    return df

# %%
df.head()

# %%
# Apply the function to add multiple indicators to the dataframe
df = add_multiple_indicators(df, ema_periods=[10, 20], atr_periods=[10, 20])

# %%
df

# %% [markdown]
# ## Find Long and Short Trades

# %%
# Create a function to identify long trade setups
def find_long_trades(df, 
                     ema_period=10, 
                     atr_period=10, 
                     target_atr_multiplier=0.5, 
                     stop_loss_atr_multiplier=1.0, 
                     tick_size=0.1):
    """
    Identify long trade setups based on the following conditions:
    - Current candle has lower high and lower low than previous candle
    - Current candle is bullish (close > open)
    - Close is above EMA
    - Entry is 1 tick above the high of the current candle
    - Target is target_atr_multiplier * ATR above entry
    - Stop loss is stop_loss_atr_multiplier * ATR below entry

    Parameters:
    df (DataFrame): DataFrame containing OHLC data with 'High', 'Low', 'Open', 'Close', 'EMA_{ema_period}', 'ATR_{atr_period}' columns.
    
    ema_period (int): Period for EMA calculation.
    atr_period (int): Period for ATR calculation.
    target_atr_multiplier: ATR multiplier for target price calculation (default: 0.5)
    stop_loss_atr_multiplier: ATR multiplier for stop loss calculation (default: 1.0)
    tick_size (float): Smallest price movement for the instrument.
    
    Returns:
    DataFrame: DataFrame containing trade setups with entry, target, stop loss, and other relevant columns.
    """
    
    import pandas as pd
    
    # Ensure the dataframe has the necessary columns
    required_columns = ['High', 'Low', 'Open', 'Close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain {required_columns} columns.")
    
    # Check if data has ema and atr period columns available
    ema_col = f'EMA_{ema_period}'
    atr_col = f'ATR_{atr_period}'
    if ema_col not in df.columns or atr_col not in df.columns:
        df = add_multiple_indicators(df, ema_periods=[ema_period], atr_periods=[atr_period])
        # raise ValueError(f"DataFrame must contain EMA_{ema_period} and ATR_{atr_period} columns.")
    
    
    # Create a copy to avoid modifying the original dataframe
    trades_df = df.copy()
    
    # Calculate if the current candle has lower high and lower low than previous candle
    trades_df['lower_high'] = trades_df['High'] < trades_df['High'].shift(1)
    trades_df['lower_low'] = trades_df['Low'] < trades_df['Low'].shift(1)
    
    # Calculate if the candle is bullish (close > open)
    trades_df['is_bullish'] = trades_df['Close'] > trades_df['Open']
    
    # Check if close is above EMA
    trades_df['above_ema'] = trades_df['Close'] > trades_df[ema_col]
    
    # Identify trade setups - when all conditions are met
    trades_df['long_setup'] = (trades_df['lower_high'] & 
                              trades_df['lower_low'] & 
                              trades_df['is_bullish'] & 
                              trades_df['above_ema'])
    
    # Calculate entry, target and stop loss for identified setups
    # Candle entry is 1 tick above the high of the current candle
    trades_df['entry'] = trades_df['High'] + tick_size  # 1 tick above high
    trades_df['target'] = trades_df['entry'] + target_atr_multiplier * trades_df[atr_col]
    trades_df['stop_loss'] = trades_df['entry'] - stop_loss_atr_multiplier * trades_df[atr_col]
    
    # Filter only the trade setups
    long_trades = trades_df[trades_df['long_setup']]
    
    return df, long_trades[['entry', 'target', 'stop_loss', 'Close', 'High', 'Low', 'Open', ema_col, atr_col]]
   


# %%
def find_short_trades(df, 
    ema_period=10, 
    atr_period=10, 
    target_atr_multiplier=0.5, 
    stop_loss_atr_multiplier=1.0, 
    tick_size=0.1):
    """
    Identify short trade setups based on the following conditions:
    - Current candle has higher high and higher low than previous candle
    - Current candle is bearish (close < open)
    - Close is below EMA
    - Entry is 1 tick below the low of the current candle
    - Target is target_atr_multiplier * ATR below entry
    - Stop loss is stop_loss_atr_multiplier * ATR above entry

    
    Parameters:
    df (DataFrame): DataFrame containing OHLC data with 'High', 'Low', 'Open', 'Close', 'EMA_{ema_period}', 'ATR_{atr_period}' columns.

    ema_period (int): Period for EMA calculation.
    atr_period (int): Period for ATR calculation.
    target_atr_multiplier: ATR multiplier for target price calculation (default: 0.5)
    stop_loss_atr_multiplier: ATR multiplier for stop loss calculation (default: 1.0)
    tick_size (float): Smallest price movement for the instrument.
    
    Returns:
    DataFrame: DataFrame containing trade setups with entry, target, stop loss, and other relevant columns.
    """
    
    # Ensure the dataframe has the necessary columns
    required_columns = ['High', 'Low', 'Open', 'Close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain {required_columns} columns.")
    
    # Check if data has ema and atr period columns available
    ema_col = f'EMA_{ema_period}'
    atr_col = f'ATR_{atr_period}'
    if ema_col not in df.columns or atr_col not in df.columns:
        df = add_multiple_indicators(df, ema_periods=[ema_period], atr_periods=[atr_period]) # Add the missing indicators to df
        # raise ValueError(f"DataFrame must contain {ema_col} and {atr_col} columns.")
    
    # Create a copy to avoid modifying the original dataframe
    trades_df = df.copy()
    
    # Calculate if the current candle has higher high and higher low than previous candle
    trades_df['higher_high'] = trades_df['High'] > trades_df['High'].shift(1)
    trades_df['higher_low'] = trades_df['Low'] > trades_df['Low'].shift(1)
    
    # Calculate if the candle is bearish (close < open)
    trades_df['is_bearish'] = trades_df['Close'] < trades_df['Open']
    
    # Check if close is below EMA
    trades_df['below_ema'] = trades_df['Close'] < trades_df[ema_col]
    
    # Identify trade setups - when all conditions are met
    trades_df['short_setup'] = (trades_df['higher_high'] & 
                              trades_df['higher_low'] & 
                              trades_df['is_bearish'] & 
                              trades_df['below_ema'])
    
    # Calculate entry, target and stop loss for identified setups
    # Candle entry is 1 tick below the low of the current candle
    trades_df['entry'] = trades_df['Low'] - tick_size  # 1 tick below low
    trades_df['target'] = trades_df['entry'] - target_atr_multiplier * trades_df[atr_col]  # Target below entry
    trades_df['stop_loss'] = trades_df['entry'] + stop_loss_atr_multiplier * trades_df[atr_col]  # Stop loss above entry
    
    # Filter only the trade setups
    short_trades = trades_df[trades_df['short_setup']]
    
    return df, short_trades[['entry', 'target', 'stop_loss', 'Close', 'High', 'Low', 'Open', ema_col, atr_col]]

# %%
# Find long trade setups
df, long_trade_opportunities = find_long_trades(df)
print(f"Found {len(long_trade_opportunities)} long trade opportunities")
long_trade_opportunities.head(10)

# %%
# Find short trade setups
df, short_trade_opportunities = find_short_trades(df)
print(f"Found {len(short_trade_opportunities)} short trade opportunities")
short_trade_opportunities.head(10)

# %%
# In long or short trade setups, drop any rows that have missing values
long_trade_opportunities.dropna(inplace=True)
short_trade_opportunities.dropna(inplace=True)

# %%
long_trade_opportunities

# %%
short_trade_opportunities

# %% [markdown]
# ## Backtesting

# %%
def calculate_max_drawdown(equity_curve):
    """
    Calculate the maximum drawdown percentage from an equity curve
    
    Parameters:
    - equity_curve: Series containing account balance over time
    
    Returns:
    - Maximum drawdown as a percentage
    """
    # Calculate the running maximum
    running_max = equity_curve.cummax()
    
    # Calculate drawdown in percentage terms
    drawdown = ((running_max - equity_curve) / running_max * 100)
    
    # Find the maximum drawdown
    max_dd = drawdown.max()
    
    return max_dd


# %%
def backtest_short_trade_strategy(df, trades_df, 
                                  slippage_pct=0.05, 
                                  commission_pct=0.075, 
                                  risk_per_trade_pct=1.0, 
                                  target_atr_multiplier=0.5, 
                                  stop_loss_atr_multiplier=1.0):
    """
    Backtests a short trading strategy on identified trade setups
    
    Parameters:
    - df: DataFrame containing full price history with OHLC data
    - trades_df: DataFrame containing identified short trade setups with entry, target and stop-loss prices
    - slippage_pct: Percentage of slippage applied to entries and exits (default: 0.05%)
    - commission_pct: Percentage of commission applied to entries and exits (default: 0.075%)
    - risk_per_trade_pct: Percentage of account balance risked per trade (default: 1%)
    - target_atr_multiplier: ATR multiplier for target price calculation (default: 0.5)
    - stop_loss_atr_multiplier: ATR multiplier for stop loss calculation (default: 1.0)
    
    Returns:
    - DataFrame containing detailed results of each trade
    - Dictionary containing aggregated performance metrics
    - DataFrame containing equity curve data
    """
    # Initialize performance tracking variables
    initial_balance = 10000.0
    balance = initial_balance
    balance_history = [initial_balance]
    dates_history = [df.index[0]]  # Start with the first date
    closed_trades = []
    
    # Loop through each trading opportunity chronologically
    dates = trades_df.index.sort_values().tolist()
    
    for trade_date in dates:
        # Extract trade setup information
        setup = trades_df.loc[trade_date]
        
        # Use the pre-calculated values from trades_df
        entry_price = setup['entry']
        target_price = setup['target']
        stop_loss_price = setup['stop_loss']

        # if either entry, target or stop loss is NaN, skip this trade
        if pd.isna(entry_price) or pd.isna(target_price) or pd.isna(stop_loss_price):
            print(f"Skipping trade on {trade_date} due to NaN entry, target or stop loss.")
            continue
        
        # Calculate position size based on risk percentage
        risk_amount = balance * (risk_per_trade_pct / 100)
        risk_per_coin = stop_loss_price - entry_price  
        
        # Avoid division by zero or negative risk
        if risk_per_coin <= 0:
            continue
            
        position_size = risk_amount / risk_per_coin
        position_value = position_size * entry_price
        
        # Check if we have enough balance for this trade
        if position_value > balance:
            position_size = balance / entry_price
            position_value = balance
        
        # Apply slippage to entry price (worse price for short trades)
        actual_entry = entry_price * (1 - slippage_pct/100)  # Lower price for short entry
        
        # Apply commission on entry
        commission = position_value * commission_pct/100
        balance -= commission
        
        # Track the trade
        trade = {
            'entry_date': trade_date,
            'entry_price': actual_entry,
            'position_size': position_size,
            'position_value': position_value,
            'target_price': target_price,
            'stop_loss': stop_loss_price,
            'commission': commission,
            'risk_amount': risk_amount,
            'risk_per_trade_pct': risk_per_trade_pct
        }
        
        # Find future candles to determine outcome
        future_dates = df.loc[trade_date:].index[1:]  # Get dates after the setup
        exit_found = False
        
        for future_date in future_dates:
            future_bar = df.loc[future_date]

            # Check if target was hit (price moved down for shorts)
            if future_bar['Low'] <= target_price:
                # Target hit - calculate exit with slippage
                actual_exit = target_price * (1 + slippage_pct/100)  # Worse price for exit (slippage)
                exit_position_value = position_size * actual_exit
                profit = position_value - exit_position_value  # Profit calculation for shorts
                
                # Apply commission on exit
                exit_commission = exit_position_value * commission_pct/100
                profit -= exit_commission
                
                # Update balance
                balance += profit
                
                # Log the trade
                trade['exit_date'] = future_date
                trade['exit_price'] = actual_exit
                trade['exit_type'] = 'target'
                trade['profit_loss'] = profit
                trade['return_pct'] = (profit / position_value) * 100
                trade['balance'] = balance
                trade['exit_commission'] = exit_commission
                trade['hold_period'] = (future_date - trade_date).total_seconds() / 3600  # Hold period in hours
                closed_trades.append(trade)
                exit_found = True
                
                # Update equity curve
                balance_history.append(balance)
                dates_history.append(future_date)
                break
            
            # Check if stop loss was hit (price moved up for shorts)
            elif future_bar['High'] >= stop_loss_price:
                # Stop loss hit - calculate exit with slippage
                actual_exit = stop_loss_price * (1 + slippage_pct/100)  # Worse price for exit (slippage)
                exit_position_value = position_size * actual_exit
                loss = position_value - exit_position_value  # Loss calculation for shorts
                
                # Apply commission on exit
                exit_commission = exit_position_value * commission_pct/100
                loss -= exit_commission
                
                # Update balance
                balance += loss
                
                # Log the trade
                trade['exit_date'] = future_date
                trade['exit_price'] = actual_exit
                trade['exit_type'] = 'stop_loss'
                trade['profit_loss'] = loss
                trade['return_pct'] = (loss / position_value) * 100
                trade['balance'] = balance
                trade['exit_commission'] = exit_commission
                trade['hold_period'] = (future_date - trade_date).total_seconds() / 3600  # Hold period in hours
                closed_trades.append(trade)
                exit_found = True
                
                # Update equity curve
                balance_history.append(balance)
                dates_history.append(future_date)
                break
        
        # If trade is still open at the end of data, close it at the last price
        if not exit_found:
            last_date = df.index[-1]
            last_bar = df.loc[last_date]
            actual_exit = last_bar['Close'] * (1 + slippage_pct/100)  # Apply slippage on exit
            exit_position_value = position_size * actual_exit
            pnl = position_value - exit_position_value  # P&L calculation for shorts
            
            # Apply commission on exit
            exit_commission = exit_position_value * commission_pct/100
            pnl -= exit_commission
            
            # Update balance
            balance += pnl
            
            # Log the trade
            trade['exit_date'] = last_date
            trade['exit_price'] = actual_exit
            trade['exit_type'] = 'end_of_data'
            trade['profit_loss'] = pnl
            trade['return_pct'] = (pnl / position_value) * 100
            trade['balance'] = balance
            trade['exit_commission'] = exit_commission
            trade['hold_period'] = (last_date - trade_date).total_seconds() / 3600  # Hold period in hours
            closed_trades.append(trade)
            
            # Update equity curve
            balance_history.append(balance)
            dates_history.append(last_date)
    
    # Create equity curve dataframe
    equity_df = pd.DataFrame({
        'date': dates_history,
        'balance': balance_history
    })
    equity_df.set_index('date', inplace=True)
    
    # Calculate strategy metrics if trades were executed
    trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()
    if len(trades_df) > 0:
        # Separate winning and losing trades
        winning_trades = trades_df[trades_df['profit_loss'] > 0]
        losing_trades = trades_df[trades_df['profit_loss'] <= 0]
        
        # Calculate max drawdown using equity curve
        max_dd = calculate_max_drawdown(equity_df['balance'])
        
        # Calculate average trade metrics
               
        metrics = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': balance - initial_balance,
            'total_return_pct': (balance / initial_balance - 1) * 100,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            'average_win': winning_trades['profit_loss'].mean() if len(winning_trades) > 0 else 0,
            'average_loss': losing_trades['profit_loss'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': winning_trades['profit_loss'].max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades['profit_loss'].min() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades['profit_loss'].sum() / losing_trades['profit_loss'].sum()) if len(losing_trades) > 0 and losing_trades['profit_loss'].sum() != 0 else float('inf'),
            'max_drawdown_pct': max_dd,
            'avg_trade_duration_hours': trades_df['hold_period'].mean() if 'hold_period' in trades_df.columns else 0,
            'total_commission': trades_df['commission'].sum() + trades_df['exit_commission'].sum() if 'exit_commission' in trades_df.columns else trades_df['commission'].sum(),
            'expectancy': (winning_trades['profit_loss'].mean() * len(winning_trades) + losing_trades['profit_loss'].mean() * len(losing_trades)) / len(trades_df) if len(trades_df) > 0 else 0
        }
    else:
        metrics = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': 0,
            'total_return_pct': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'average_win': 0,
            'average_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'profit_factor': 0,
            'max_drawdown_pct': 0,
            'avg_trade_duration_hours': 0,
            'total_commission': 0,
            'expectancy': 0
        }
    
    return trades_df, metrics, equity_df

# %%
def backtest_long_trade_strategy(df, trades_df, 
                                slippage_pct=0.05, 
                                commission_pct=0.075, 
                                risk_per_trade_pct=1.0, 
                                target_atr_multiplier=0.5, 
                                stop_loss_atr_multiplier=1.0):
    """
    Backtests a long trading strategy on identified trade setups
    
    Parameters:
    - df: DataFrame containing full price history with OHLC data
    - trades_df: DataFrame containing identified long trade setups with entry, target and stop-loss prices
    - slippage_pct: Percentage of slippage applied to entries and exits (default: 0.05%)
    - commission_pct: Percentage of commission applied to entries and exits (default: 0.075%)
    - risk_per_trade_pct: Percentage of account balance risked per trade (default: 1%)
    - target_atr_multiplier: ATR multiplier for target price calculation (default: 0.5)
    - stop_loss_atr_multiplier: ATR multiplier for stop loss calculation (default: 1.0)
    
    Returns:
    - DataFrame containing detailed results of each trade
    - Dictionary containing aggregated performance metrics
    - DataFrame containing equity curve data
    """
    # Initialize performance tracking variables
    initial_balance = 10000.0
    balance = initial_balance
    balance_history = [initial_balance]
    dates_history = [df.index[0]]  # Start with the first date
    closed_trades = []
    
    # Loop through each trading opportunity chronologically
    dates = trades_df.index.sort_values().tolist()
    
    for trade_date in dates:
        # Extract trade setup information
        setup = trades_df.loc[trade_date]
        
        # Use the pre-calculated values from trades_df
        entry_price = setup['entry']
        target_price = setup['target']
        stop_loss_price = setup['stop_loss']

        # if either entry, target or stop loss is NaN, skip this trade
        if pd.isna(entry_price) or pd.isna(target_price) or pd.isna(stop_loss_price):
            print(f"Skipping trade on {trade_date} due to NaN entry, target or stop loss.")
            continue
        
        # Calculate position size based on risk percentage
        risk_amount = balance * (risk_per_trade_pct / 100)
        risk_per_coin = entry_price - stop_loss_price  # For long trades, risk is entry minus stop loss
        
        # Avoid division by zero or negative risk
        if risk_per_coin <= 0:
            continue
            
        position_size = risk_amount / risk_per_coin
        position_value = position_size * entry_price
        
        # Check if we have enough balance for this trade
        if position_value > balance:
            position_size = balance / entry_price
            position_value = balance
        
        # Apply slippage to entry price (worse price for long trades)
        actual_entry = entry_price * (1 + slippage_pct/100)  # Higher price for long entry
        
        # Apply commission on entry
        commission = position_value * commission_pct/100
        balance -= position_value + commission  # Deduct the position value plus commission
        
        # Track the trade
        trade = {
            'entry_date': trade_date,
            'entry_price': actual_entry,
            'position_size': position_size,
            'position_value': position_value,
            'target_price': target_price,
            'stop_loss': stop_loss_price,
            'commission': commission,
            'risk_amount': risk_amount,
            'risk_per_trade_pct': risk_per_trade_pct
        }
        
        # Find future candles to determine outcome
        future_dates = df.loc[trade_date:].index[1:]  # Get dates after the setup
        exit_found = False
        
        for future_date in future_dates:
            future_bar = df.loc[future_date]

            # Check if target was hit (price moved up for longs)
            if future_bar['High'] >= target_price:
                # Target hit - calculate exit with slippage
                actual_exit = target_price * (1 - slippage_pct/100)  # Worse price for exit (slippage)
                exit_position_value = position_size * actual_exit
                
                # Apply commission on exit
                exit_commission = exit_position_value * commission_pct/100
                
                # Calculate profit (exit value minus entry value minus commissions)
                profit = exit_position_value - position_value - exit_commission
                
                # Update balance
                balance += exit_position_value - exit_commission
                
                # Log the trade
                trade['exit_date'] = future_date
                trade['exit_price'] = actual_exit
                trade['exit_type'] = 'target'
                trade['profit_loss'] = profit
                trade['return_pct'] = (profit / position_value) * 100
                trade['balance'] = balance
                trade['exit_commission'] = exit_commission
                trade['hold_period'] = (future_date - trade_date).total_seconds() / 3600  # Hold period in hours
                closed_trades.append(trade)
                exit_found = True
                
                # Update equity curve
                balance_history.append(balance)
                dates_history.append(future_date)
                break
            
            # Check if stop loss was hit (price moved down for longs)
            elif future_bar['Low'] <= stop_loss_price:
                # Stop loss hit - calculate exit with slippage
                actual_exit = stop_loss_price * (1 - slippage_pct/100)  # Worse price for exit (slippage)
                exit_position_value = position_size * actual_exit
                
                # Apply commission on exit
                exit_commission = exit_position_value * commission_pct/100
                
                # Calculate loss (exit value minus entry value minus commissions)
                loss = exit_position_value - position_value - exit_commission
                
                # Update balance
                balance += exit_position_value - exit_commission
                
                # Log the trade
                trade['exit_date'] = future_date
                trade['exit_price'] = actual_exit
                trade['exit_type'] = 'stop_loss'
                trade['profit_loss'] = loss
                trade['return_pct'] = (loss / position_value) * 100
                trade['balance'] = balance
                trade['exit_commission'] = exit_commission
                trade['hold_period'] = (future_date - trade_date).total_seconds() / 3600  # Hold period in hours
                closed_trades.append(trade)
                exit_found = True
                
                # Update equity curve
                balance_history.append(balance)
                dates_history.append(future_date)
                break
        
        # If trade is still open at the end of data, close it at the last price
        if not exit_found:
            last_date = df.index[-1]
            last_bar = df.loc[last_date]
            actual_exit = last_bar['Close'] * (1 - slippage_pct/100)  # Apply slippage on exit
            exit_position_value = position_size * actual_exit
            
            # Apply commission on exit
            exit_commission = exit_position_value * commission_pct/100
            
            # Calculate P&L (exit value minus entry value minus commissions)
            pnl = exit_position_value - position_value - exit_commission
            
            # Update balance
            balance += exit_position_value - exit_commission
            
            # Log the trade
            trade['exit_date'] = last_date
            trade['exit_price'] = actual_exit
            trade['exit_type'] = 'end_of_data'
            trade['profit_loss'] = pnl
            trade['return_pct'] = (pnl / position_value) * 100
            trade['balance'] = balance
            trade['exit_commission'] = exit_commission
            trade['hold_period'] = (last_date - trade_date).total_seconds() / 3600  # Hold period in hours
            closed_trades.append(trade)
            
            # Update equity curve
            balance_history.append(balance)
            dates_history.append(last_date)
    
    # Create equity curve dataframe
    equity_df = pd.DataFrame({
        'date': dates_history,
        'balance': balance_history
    })
    equity_df.set_index('date', inplace=True)
    
    # Calculate strategy metrics if trades were executed
    trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()
    if len(trades_df) > 0:
        # Separate winning and losing trades
        winning_trades = trades_df[trades_df['profit_loss'] > 0]
        losing_trades = trades_df[trades_df['profit_loss'] <= 0]
        
        # Calculate max drawdown using equity curve
        max_dd = calculate_max_drawdown(equity_df['balance'])
        
        # Calculate average trade metrics
        metrics = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': balance - initial_balance,
            'total_return_pct': (balance / initial_balance - 1) * 100,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            'average_win': winning_trades['profit_loss'].mean() if len(winning_trades) > 0 else 0,
            'average_loss': losing_trades['profit_loss'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': winning_trades['profit_loss'].max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades['profit_loss'].min() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades['profit_loss'].sum() / losing_trades['profit_loss'].sum()) if len(losing_trades) > 0 and losing_trades['profit_loss'].sum() != 0 else float('inf'),
            'max_drawdown_pct': max_dd,
            'avg_trade_duration_hours': trades_df['hold_period'].mean() if 'hold_period' in trades_df.columns else 0,
            'total_commission': trades_df['commission'].sum() + trades_df['exit_commission'].sum() if 'exit_commission' in trades_df.columns else trades_df['commission'].sum(),
            'expectancy': (winning_trades['profit_loss'].mean() * len(winning_trades) + losing_trades['profit_loss'].mean() * len(losing_trades)) / len(trades_df) if len(trades_df) > 0 else 0
        }
    else:
        metrics = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': 0,
            'total_return_pct': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'average_win': 0,
            'average_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'profit_factor': 0,
            'max_drawdown_pct': 0,
            'avg_trade_duration_hours': 0,
            'total_commission': 0,
            'expectancy': 0
        }
    
    return trades_df, metrics, equity_df

# %% [markdown]
# Get the results from backtesting

# %%
# Run backtest for short trades
short_trade_results, short_metrics, short_equity_curve = backtest_short_trade_strategy(
    df, 
    short_trade_opportunities,
    slippage_pct=0.05,     # 0.05% slippage
    commission_pct=0.075,  # 0.075% commission (Bybit's standard taker fee)
    risk_per_trade_pct=1.0,  # Risk 1% of account per trade
    target_atr_multiplier=0.5,  # Target is 0.5x ATR
    stop_loss_atr_multiplier=1.0  # Stop loss is 1x ATR
)

# %%
# Run backtest for long trades
long_trade_results, long_metrics, long_equity_curve = backtest_long_trade_strategy(
    df, 
    long_trade_opportunities,
    slippage_pct=0.05,     # 0.05% slippage
    commission_pct=0.075,  # 0.075% commission (Bybit's standard taker fee)
    risk_per_trade_pct=1.0,  # Risk 1% of account per trade
    target_atr_multiplier=0.5,  # Target is 0.5x ATR
    stop_loss_atr_multiplier=1.0  # Stop loss is 1x ATR
)


# %%
long_trade_results

# %%
long_metrics

# %%
long_equity_curve

# %% [markdown]
# ## Cryptocurrency Trading Strategy Backtesting Analysis
# =====================================================
# 
# This backtest simulation implements a comprehensive framework for evaluating trading strategies on the cryptocurrency market. The analysis follows a systematic approach to simulate real-world trading conditions with precise entry/exit mechanics and risk management protocols.
# 
# **Backtesting Methodology**
# ---
# 
# **1. Trade Setup and Entry Logic**
# *   **Long Trade Setup**: Identifies bullish setups where a candle has a lower high and lower low than the previous candle, is bullish (close > open), and closes above a key moving average (EMA).
# *   **Short Trade Setup**: Identifies bearish setups where a candle has a higher high and higher low than the previous candle, is bearish (close < open), and closes below a key moving average (EMA).
# *   **Entry Points**: For each trade setup identified, the simulation enters at a precise entry price (e.g., for long trades, 1 tick above the setup candle's high).
# 
# **2. Risk Management and Position Sizing**
# *   **Fixed Percentage Risk**: Each trade risks exactly 1% of the current account balance.
# *   **Position Size Calculation**: The size of each position is calculated based on the risk amount and the distance between the entry price and the stop loss.
# 
#     `Position Size = (Account Balance * 0.01) / Distance between Entry and Stop Loss`
# 
# *   **Dynamic Scaling**: As the account equity grows or shrinks, the position size for subsequent trades adjusts accordingly.
# *   **Predetermined Exits**: Both take profit and stop loss levels are determined at the time of entry, often based on a multiplier of the Average True Range (ATR).
# 
# **3. Trade Execution and Realistic Costs**
# *   **Slippage Modeling**: Applies a realistic 0.05% slippage on both entries and exits to simulate the difference between the expected price and the actual fill price.
# *   **Commission Structure**: Applies a 0.075% commission fee on both the entry and exit of a trade to account for transaction costs.
# 
# **4. Trade Management and Exit Mechanics**
# Once a trade is entered, it remains open until one of the following exit conditions is met:
# *   **Target Price Hit**: The trade is closed for a profit when the price reaches the predefined take profit level.
# *   **Stop Loss Hit**: The trade is closed for a loss when the price reaches the predefined stop loss level.
# *   **End of Data**: Any open trades at the end of the testing period are closed at the final available market price.
# 
# **5. Performance Measurement and Tracking**
# *   **Equity Curve**: Records the account balance after every closed trade to visualize performance and growth over time.
# *   **Detailed Trade Log**: Maintains a comprehensive log for every trade, including entry/exit dates, prices, and the final profit or loss.
# *   **Key Performance Metrics**: Calculates essential metrics to judge the strategy's effectiveness, including:
#     *   Win Rate
#     *   Profit Factor (Gross Profit / Gross Loss)
#     *   Maximum Drawdown
#     *   Average Trade Duration
# 
# **Strengths of this Strategy**
# *   Well-defined and systematic entry and exit rules.
# *   Integrated risk management through fixed-percentage position sizing.
# *   Realistic simulation that accounts for trading friction like slippage and commissions.
# *   Clear and comprehensive performance metrics for objective evaluation.
# 
# **Limitations and Potential Improvements**
# - No trailing stop loss mechanism to capture larger moves
# - Fixed target based only on ATR multiplier (could be optimized)
# - No filter for high impact news events or market conditions
# - Could implement pyramiding (adding to winning positions)
# - Could add filters based on higher timeframes for better entry timing

# %% [markdown]
# ## Results

# %% [markdown]
# ### Trade Statistics

# %%
import numpy as np
import pandas as pd

def analyze_and_display_trade_statistics(trade_results=None, metrics=None, trade_type="Long"):
    """
    Display summary statistics and perform deeper analysis for a trading strategy.
    
    Parameters:
    - trade_results: DataFrame containing detailed results of each trade
    - metrics: Dictionary containing aggregated performance metrics
    - trade_type: String indicating the type of trades ("Long" or "Short")
    
    Returns:
    - None (prints statistics summary and analysis to console)
    """
    import matplotlib.pyplot as plt
    
    if trade_results is None or metrics is None:
        print(f"No {trade_type.lower()} trade data provided.")
        return
        
    # Display basic statistics
    print(f"===== {trade_type.upper()} TRADE BACKTEST SUMMARY =====")
    print(f"Initial Balance: ${metrics['initial_balance']:,.2f}")
    print(f"Final Balance: ${metrics['final_balance']:,.2f}")
    print(f"Total Return: ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")
    
    # Trade statistics
    print(f"\n----- Trade Statistics -----")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']} ({metrics['win_rate']*100:.2f}%)")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Average Win: ${metrics['average_win']:.2f}")
    print(f"Average Loss: ${metrics['average_loss']:.2f}")
    print(f"Largest Win: ${metrics['largest_win']:.2f}")
    print(f"Largest Loss: ${metrics['largest_loss']:.2f}")
    
    # Risk and duration metrics
    print(f"\n----- Risk Metrics -----")
    print(f"Maximum Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Average Trade Duration: {metrics['avg_trade_duration_hours']:.2f} hours")
    print(f"Total Commission Paid: ${metrics['total_commission']:.2f}")
    
    # Calculate additional metrics if we have trade data
    if len(trade_results) > 0:
        # Win/loss ratio
        win_loss_ratio = abs(metrics['average_win'] / metrics['average_loss']) if metrics['average_loss'] != 0 else float('inf')
        
        # Expectancy and Sharpe ratio
        expectancy = metrics['win_rate'] * metrics['average_win'] + (1 - metrics['win_rate']) * metrics['average_loss']
        risk_reward_ratio = abs(metrics['average_win'] / metrics['average_loss']) if metrics['average_loss'] != 0 else float('inf')
        
        # Win streaks
        trade_results['is_win'] = trade_results['profit_loss'] > 0
        trade_results['streak_change'] = trade_results['is_win'] != trade_results['is_win'].shift(1)
        trade_results['streak_id'] = trade_results['streak_change'].cumsum()
        streaks = trade_results.groupby(['streak_id', 'is_win']).size()
        
        max_win_streak = streaks[streaks.index.get_level_values('is_win') == True].max() if True in streaks.index.get_level_values('is_win') else 0
        max_loss_streak = streaks[streaks.index.get_level_values('is_win') == False].max() if False in streaks.index.get_level_values('is_win') else 0
        
    
        print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
        print(f"Expectancy per Trade: ${expectancy:.2f}")
        print(f"Risk-Reward Ratio: {risk_reward_ratio:.2f}")
        print(f"Maximum Win Streak: {max_win_streak}")
        print(f"Maximum Loss Streak: {max_loss_streak}")
        
        # Exit type distribution
        exit_counts = trade_results['exit_type'].value_counts()
        print(f"\n----- Exit Types -----")
        for exit_type, count in exit_counts.items():
            print(f"{exit_type}: {count} ({count/len(trade_results)*100:.1f}%)")
    
        # Time-based analysis
        trade_results['duration'] = (trade_results['exit_date'] - trade_results['entry_date'])
        trade_results['duration_hours'] = trade_results['duration'].dt.total_seconds() / 3600
        
        # Extract time components
        trade_results['hour'] = trade_results['entry_date'].dt.hour
        trade_results['day_of_week'] = trade_results['entry_date'].dt.dayofweek
        
        # Create bins for trade duration
        duration_bins = [0, 1, 4, 8, 24, 48, float('inf')]
        duration_labels = ['<1h', '1-4h', '4-8h', '8-24h', '1-2d', '>2d']
        trade_results['duration_group'] = pd.cut(trade_results['duration_hours'], 
                                                bins=duration_bins, 
                                                labels=duration_labels)
        
        # Group trades by duration - fixing the FutureWarning by adding observed=True
        duration_analysis = trade_results.groupby('duration_group', observed=True).agg({
            'profit_loss': ['count', 'mean', 'sum'],
            'return_pct': ['mean', 'median']
        })
        
        # Group trades by hour of day
        hour_analysis = trade_results.groupby('hour').agg({
            'profit_loss': ['count', 'mean', 'sum'],
            'return_pct': 'mean'
        })
        
        # Group trades by day of week
        day_analysis = trade_results.groupby('day_of_week').agg({
            'profit_loss': ['count', 'mean', 'sum'],
            'return_pct': 'mean'
        })
        
        # Calculate additional risk metrics
        sharpe_ratio = (metrics['total_return_pct'] / 100) / (trade_results['return_pct'].std() / 100 * np.sqrt(252)) if trade_results['return_pct'].std() != 0 else 0
        expectancy_per_dollar = expectancy / abs(metrics['average_loss']) if metrics['average_loss'] != 0 else 0
        
        # Create a dictionary of additional metrics
        additional_metrics = {
            'duration_analysis': duration_analysis,
            'hour_analysis': hour_analysis,
            'day_analysis': day_analysis,
            'sharpe_ratio': sharpe_ratio,
            'expectancy_per_dollar': expectancy_per_dollar
        }
        
        print(f"\n----- Time-Based Analysis -----")

        print("\n===== DURATION ANALYSIS =====")
        print(additional_metrics['duration_analysis'])

        print("\n===== HOUR ANALYSIS =====")
        print(additional_metrics['hour_analysis'])

        print("\n===== DAY ANALYSIS =====")
        print(additional_metrics['day_analysis'])

        print("\nPerformance by Trade Duration:")
        print(duration_analysis['profit_loss']['mean'].to_string())
        
        print("\nPerformance by Hour of Day (Top 3 and Bottom 3):")
        hour_performance = hour_analysis['profit_loss']['mean'].sort_values(ascending=False)
        print("Best Hours:")
        print(hour_performance.head(3).to_string())
        print("\nWorst Hours:")
        print(hour_performance.tail(3).to_string())
        
        print(f"\n----- Monthly Distribution -----")
        if 'month' not in trade_results.columns:
            trade_results['month'] = trade_results['entry_date'].dt.month
            
        monthly_distribution = trade_results.groupby('month').agg({
            'profit_loss': ['count', 'sum', 'mean'],
            'is_win': 'mean'
        })
        
        print("\nTop 3 Best Months:")
        best_months = monthly_distribution['profit_loss']['mean'].sort_values(ascending=False).head(3)
        for month, value in best_months.items():
            print(f"Month {month}: ${value:.2f} avg profit/loss")
            
        # Display top 5 profitable trades
        print(f"\n----- Top 5 Profitable Trades -----")
        display_cols = ['entry_date', 'exit_date', 'entry_price', 'exit_price', 
                       'exit_type', 'profit_loss', 'return_pct', 'hold_period']
        top_trades = trade_results[display_cols].sort_values('profit_loss', ascending=False).head(5)
        print(top_trades)
        
        print(f"\n----- Worst 5 Losing Trades -----")
        worst_trades = trade_results[display_cols].sort_values('profit_loss').head(5)
        print(worst_trades)
        
        # Additional system quality metrics
        print(f"\n----- System Quality Metrics -----")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Expectancy per Dollar Risked: {expectancy_per_dollar:.2f}")
        
        # Return distribution statistics
        returns_mean = trade_results['return_pct'].mean()
        returns_std = trade_results['return_pct'].std()
        returns_skew = trade_results['return_pct'].skew()
        returns_kurt = trade_results['return_pct'].kurtosis()
        
        print(f"\n----- Return Distribution -----")
        print(f"Mean Return: {returns_mean:.3f}%")
        print(f"Standard Deviation: {returns_std:.3f}%")
        print(f"Skewness: {returns_skew:.3f}")
        print(f"Kurtosis: {returns_kurt:.3f}")


        
        # System reliability metric
        reliability = metrics['win_rate'] * win_loss_ratio if metrics['win_rate'] > 0 else 0
        print(f"System Reliability Score: {reliability:.3f}")


# %% [markdown]
# #### Long Trade Stats

# %%
analyze_and_display_trade_statistics(trade_results=long_trade_results, metrics=long_metrics, trade_type="Long")

# %% [markdown]
# #### Short Trade Stats

# %%
analyze_and_display_trade_statistics(trade_results=short_trade_results, metrics=short_metrics, trade_type="Short")

# %% [markdown]
# ### Visualizations of Trade Statistics

# %%
# Function to visualize trade outcomes
def visualize_trade_outcomes(backtest_results):
    """
    Visualizes the distribution of trade outcomes, profit/loss per trade, and drawdown periods.
    
    Parameters:
    - backtest_results: DataFrame containing individual trade results from the backtest
    
    Returns:
    - None (generates plots)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    # Check if there are trades to analyze
    if len(backtest_results) == 0:
        print("No trades to visualize.")
        return
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Trade Outcome Distribution (Pie Chart)
    outcome_counts = backtest_results['exit_type'].value_counts()
    axes[0, 0].pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Trade Outcome Distribution')
    
    # 2. Profit/Loss Distribution (Histogram)
    sns.histplot(backtest_results['profit_loss'], bins=30, kde=True, ax=axes[0, 1])
    axes[0, 1].axvline(x=0, color='r', linestyle='--')
    axes[0, 1].set_title('Profit/Loss Distribution')
    axes[0, 1].set_xlabel('Profit/Loss ($)')
    
    # 3. P&L by Exit Type (Box plot)
    sns.boxplot(x='exit_type', y='profit_loss', data=backtest_results, ax=axes[1, 0])
    axes[1, 0].set_title('P&L by Exit Type')
    axes[1, 0].set_xlabel('Exit Type')
    axes[1, 0].set_ylabel('Profit/Loss ($)')
    
    # 4. Cumulative P&L Over Time
    backtest_results.sort_values('exit_date', inplace=True)
    backtest_results['cumulative_pnl'] = backtest_results['profit_loss'].cumsum()
    backtest_results.plot(x='exit_date', y='cumulative_pnl', ax=axes[1, 1], legend=False)
    axes[1, 1].set_title('Cumulative P&L Over Time')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Cumulative P&L ($)')
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis: Monthly performance heatmap
    if len(backtest_results) > 10:
        backtest_results['year'] = backtest_results['exit_date'].dt.year
        backtest_results['month'] = backtest_results['exit_date'].dt.month
        
        # Group by year and month to get performance metrics
        monthly_perf = backtest_results.groupby(['year', 'month']).agg({
            'profit_loss': 'sum',
            'entry_date': 'count'  # Count of trades
        }).reset_index()
        
        monthly_perf = monthly_perf.rename(columns={'entry_date': 'num_trades'})
        
        # Create pivot tables for heatmaps
        profit_pivot = monthly_perf.pivot(index='month', columns='year', values='profit_loss')
        trades_pivot = monthly_perf.pivot(index='month', columns='year', values='num_trades')
        
        # Create heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        sns.heatmap(profit_pivot, cmap='RdYlGn', center=0, annot=True, fmt='.0f', ax=axes[0])
        axes[0].set_title('Monthly Profit/Loss ($)')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Month')
        
        sns.heatmap(trades_pivot, cmap='Blues', annot=True, fmt='d', ax=axes[1])
        axes[1].set_title('Monthly Trade Count')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Month')
        
        plt.tight_layout()
        plt.show()




# %% [markdown]
# #### Display Long Trade Stats

# %%
visualize_trade_outcomes(long_trade_results)

# %% [markdown]
# #### Display Short Trade Stats

# %%
visualize_trade_outcomes(short_trade_results)

# %% [markdown]
# ### Long vs Short Trades Statistics

# %%
def visualize_trading_statistics(long_metrics=None, short_metrics=None):
    """
    Creates a visual comparison of long and short trading statistics.
    
    Parameters:
    - long_metrics: Dictionary containing performance metrics for long trades
    - short_metrics: Dictionary containing performance metrics for short trades
    
    Returns:
    - None (displays visualizations)
    """
    # Check if we have metrics to visualize
    if long_metrics is None and short_metrics is None:
        print("No trading metrics available to visualize.")
        return
    
    # Set up the figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Common metrics to compare
    metrics_to_plot = [
        ('win_rate', 'Win Rate', '%'),
        ('profit_factor', 'Profit Factor', ''),
        ('average_win', 'Average Win', '$'),
        ('average_loss', 'Average Loss', '$'),
        ('total_return_pct', 'Total Return', '%'),
        ('max_drawdown_pct', 'Max Drawdown', '%')
    ]
    
    # Create data for bar chart comparison
    labels = []
    long_values = []
    short_values = []
    
    for metric, label, _ in metrics_to_plot:
        labels.append(label)
        
        if long_metrics is not None and metric in long_metrics:
            if metric == 'win_rate':
                long_values.append(long_metrics[metric] * 100)  # Convert to percentage
            else:
                long_values.append(long_metrics[metric])
        else:
            long_values.append(0)
        
        if short_metrics is not None and metric in short_metrics:
            if metric == 'win_rate':
                short_values.append(short_metrics[metric] * 100)  # Convert to percentage
            else:
                short_values.append(short_metrics[metric])
        else:
            short_values.append(0)
    
    # 1. Bar chart comparing key metrics
    x = np.arange(len(labels))
    width = 0.35
    
    axs[0, 0].bar(x - width/2, long_values, width, label='Long Trades', color='royalblue', alpha=0.7)
    axs[0, 0].bar(x + width/2, short_values, width, label='Short Trades', color='orange', alpha=0.7)
    
    axs[0, 0].set_title('Strategy Performance Metrics Comparison', fontsize=14)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels, rotation=45, ha='right')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # 2. Pie charts for win/loss distribution
    if long_metrics is not None:
        long_win_loss = [long_metrics['winning_trades'], long_metrics['losing_trades']]
        axs[0, 1].pie(long_win_loss, labels=['Wins', 'Losses'], autopct='%1.1f%%', 
                     colors=['royalblue', 'orange'], startangle=90)
        axs[0, 1].set_title('Long Trades Win/Loss Distribution', fontsize=14)
    
    if short_metrics is not None:
        short_win_loss = [short_metrics['winning_trades'], short_metrics['losing_trades']]
        axs[1, 1].pie(short_win_loss, labels=['Wins', 'Losses'], autopct='%1.1f%%', 
                     colors=['royalblue', 'orange'], startangle=90)
        axs[1, 1].set_title('Short Trades Win/Loss Distribution', fontsize=14)
    
    # 3. Risk-reward visualization
    if long_metrics is not None and short_metrics is not None:
        # Create data points for risk-reward comparison
        strategies = ['Long Trades', 'Short Trades']
        returns = [long_metrics['total_return_pct'], short_metrics['total_return_pct']]
        drawdowns = [long_metrics['max_drawdown_pct'], short_metrics['max_drawdown_pct']]

        scatter_colors = ['royalblue', 'orange']
        for i, strat in enumerate(strategies):
            axs[1, 0].scatter(drawdowns[i], returns[i], s=300, alpha=0.7, color=scatter_colors[i], label=strat)
            axs[1, 0].annotate(strat, (drawdowns[i], returns[i]), xytext=(10, 10), 
                             textcoords='offset points', fontsize=12)
        
        axs[1, 0].set_title('Risk-Return Comparison', fontsize=14)
        axs[1, 0].set_xlabel('Maximum Drawdown (%)', fontsize=12)
        axs[1, 0].set_ylabel('Total Return (%)', fontsize=12)
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary explanation
    print("=== TRADING STATISTICS EXPLANATION ===")
    print("\nThe visualizations above show:")
    print("1. Top Left: Bar chart comparing key performance metrics between long and short trades")
    print("2. Top Right: Pie chart showing win/loss distribution for long trades")
    print("3. Bottom Left: Risk-return scatter plot comparing drawdown vs. return")
    print("4. Bottom Right: Pie chart showing win/loss distribution for short trades")
    
    print("\n=== KEY METRICS EXPLAINED ===")
    print("• Win Rate: Percentage of trades that were profitable")
    print("• Profit Factor: Ratio of gross profits to gross losses (>1 is profitable)")
    print("• Average Win: Average profit on winning trades")
    print("• Average Loss: Average loss on losing trades")
    print("• Total Return: Percentage gain/loss over the backtest period")
    print("• Max Drawdown: Largest peak-to-trough decline in account value")
    
    # Calculate and print additional analytics
    if long_metrics is not None and short_metrics is not None:
        print("\n=== STRATEGY COMPARISON ===")
        better_win_rate = "Long" if long_metrics['win_rate'] > short_metrics['win_rate'] else "Short"
        better_profit_factor = "Long" if long_metrics['profit_factor'] > short_metrics['profit_factor'] else "Short"
        better_return = "Long" if long_metrics['total_return_pct'] > short_metrics['total_return_pct'] else "Short"
        lower_drawdown = "Long" if long_metrics['max_drawdown_pct'] < short_metrics['max_drawdown_pct'] else "Short"
        
        print(f"• Better Win Rate: {better_win_rate} strategy")
        print(f"• Better Profit Factor: {better_profit_factor} strategy")
        print(f"• Better Total Return: {better_return} strategy")
        print(f"• Lower Maximum Drawdown: {lower_drawdown} strategy")

# %%
visualize_trading_statistics(long_metrics=long_metrics, short_metrics=short_metrics)

# %% [markdown]
# #### Equity Curve Plot

# %%
def plot_combined_equity_curves(long_equity_curve=None, short_equity_curve=None, plot_title = ''):
    """
    Plot equity curves for long trades and short trades with improved styling
    
    Parameters:
    - long_equity_curve: DataFrame containing long trade equity curve (optional)
    - short_equity_curve: DataFrame containing short trade equity curve (optional)
    
    Returns:
    - None (displays plots)
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    plt.figure(figsize=(14, 8))
    
    # Check if we have data for each curve
    has_long = long_equity_curve is not None and not long_equity_curve.empty
    has_short = short_equity_curve is not None and not short_equity_curve.empty
    
    # Plot individual curves with improved styling
    if has_long:
        plt.plot(long_equity_curve.index, long_equity_curve['balance'], 
                 linestyle='-', linewidth=2, color='royalblue', 
                 label='Long Strategy', alpha=0.8)
    
    if has_short:
        plt.plot(short_equity_curve.index, short_equity_curve['balance'], 
                 linestyle='--', linewidth=2, color='orange', 
                 label='Short Strategy', alpha=0.8)
    
    # Add horizontal line at initial balance
    plt.axhline(y=10000, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Initial Balance')
    
    # Add chart details
    plt.title(f'{plot_title} Trading Strategy Equity Curves', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Account Balance ($)', fontsize=12)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis with dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    # Format y-axis with dollar signs
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
    
    # Add annotations if we have data
    if has_long:
        long_final = long_equity_curve['balance'].iloc[-1]
        plt.annotate(f'Long Final: ${long_final:.2f}',
                     xy=(long_equity_curve.index[-1], long_final),
                     xytext=(10, 10), textcoords='offset points',
                     color='royalblue', fontweight='bold')
    
    if has_short:
        short_final = short_equity_curve['balance'].iloc[-1]
        plt.annotate(f'Short Final: ${short_final:.2f}',
                     xy=(short_equity_curve.index[-1], short_final),
                     xytext=(10, -20), textcoords='offset points',
                     color='orange', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# %%
plot_combined_equity_curves(long_equity_curve=long_equity_curve, short_equity_curve=short_equity_curve, plot_title='BTCUSDT')

# %% [markdown]
# #### Analysis of Seasonality Patterns

# %%
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def analyze_trading_seasonality_patterns(long_trade_results, short_trade_results):
    """
    Analyzes time-based patterns in trading performance and creates visualizations
    showing when trades perform best throughout the day/week/month.
    
    Parameters:
    - long_trade_results: DataFrame containing long trade results
    - short_trade_results: DataFrame containing short trade results
    
    Returns:
    - combined_df: DataFrame containing combined trade results for analysis
    """
    import matplotlib.pyplot as plt
    
    # Create combined dataframe for analysis
    long_df = long_trade_results.copy() if long_trade_results is not None else pd.DataFrame()
    short_df = short_trade_results.copy() if short_trade_results is not None else pd.DataFrame()
    
    if len(long_df) > 0:
        long_df['trade_type'] = 'Long'
        
    if len(short_df) > 0:
        short_df['trade_type'] = 'Short'
    
    # Combine dataframes
    combined_df = pd.concat([long_df, short_df], ignore_index=True) if len(long_df) > 0 and len(short_df) > 0 else (long_df if len(long_df) > 0 else short_df)

    # Ensure we have data to analyze
    if len(combined_df) == 0:
        print("No trade data available for analysis.")
        return
    
    # Create custom colormap for profit/loss
    colors = ["#d73027", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850"]
    n_colors = 256
    cmap = LinearSegmentedColormap.from_list("profit_loss_cmap", colors, N=n_colors)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # Extract time components
    combined_df['hour'] = combined_df['entry_date'].dt.hour
    combined_df['day_of_week'] = combined_df['entry_date'].dt.dayofweek
    combined_df['day_name'] = combined_df['entry_date'].dt.day_name()
    combined_df['month'] = combined_df['entry_date'].dt.month
    combined_df['week_of_year'] = combined_df['entry_date'].dt.isocalendar().week


    # 1. Hourly performance heatmap
    hourly_perf = combined_df.pivot_table(
        values='profit_loss', 
        index='hour',
        columns='trade_type',
        aggfunc='mean'
    ).fillna(0)
    
    sns.heatmap(hourly_perf, cmap=cmap, center=0, annot=True, fmt=".2f", ax=axes[0, 0])
    axes[0, 0].set_title('Average Profit/Loss by Hour of Day', fontsize=14)
    axes[0, 0].set_xlabel('Trade Type')
    axes[0, 0].set_ylabel('Hour of Day')
    
    # 2. Day of week performance
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # First create a categorical column
    combined_df['day_name_ordered'] = pd.Categorical(combined_df['day_name'], categories=day_order, ordered=True)
    # Then use it for pivot table
    day_perf = combined_df.pivot_table(
        values='profit_loss',
        index='day_name_ordered',
        columns='trade_type',
        aggfunc='mean',
        observed=True
    ).fillna(0)
    
    sns.heatmap(day_perf, cmap=cmap, center=0, annot=True, fmt=".2f", ax=axes[0, 1])
    axes[0, 1].set_title('Average Profit/Loss by Day of Week', fontsize=14)
    axes[0, 1].set_xlabel('Trade Type')
    axes[0, 1].set_ylabel('Day of Week')
    
    # 3. Monthly performance
    monthly_perf = combined_df.pivot_table(
        values='profit_loss',
        index='month',
        columns='trade_type',
        aggfunc='mean'
    ).fillna(0)
    
    sns.heatmap(monthly_perf, cmap=cmap, center=0, annot=True, fmt=".2f", ax=axes[0, 2])
    axes[0, 2].set_title('Average Profit/Loss by Month', fontsize=14)
    axes[0, 2].set_xlabel('Trade Type')
    axes[0, 2].set_ylabel('Month')
    
    # 4. Trade duration vs. profit/loss scatter
    for trade_type, color in zip(['Long', 'Short'], ['royalblue', 'orange']):
        if trade_type in combined_df['trade_type'].values:
            subset = combined_df[combined_df['trade_type'] == trade_type]
            axes[1, 0].scatter(
                subset['hold_period'], 
                subset['profit_loss'], 
                alpha=0.5, 
                label=trade_type,
                color=color
            )
    
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('Trade Duration vs. Profit/Loss', fontsize=14)
    axes[1, 0].set_xlabel('Hold Period (hours)')
    axes[1, 0].set_ylabel('Profit/Loss ($)')
    axes[1, 0].legend()
    
    # 5. Cumulative trades over time
    combined_df = combined_df.sort_values('entry_date')
    
    # Create separate cumulative counts for each trade type
    for trade_type in combined_df['trade_type'].unique():
        mask = combined_df['trade_type'] == trade_type
        combined_df.loc[mask, f'cumulative_{trade_type.lower()}'] = range(1, mask.sum() + 1)
    
    # Plot the cumulative trades for each type
    for trade_type, color in zip(['Long', 'Short'], ['royalblue', 'orange']):
        if trade_type in combined_df['trade_type'].values:
            subset = combined_df[combined_df['trade_type'] == trade_type]
            axes[1, 1].plot(
                subset['entry_date'], 
                subset[f'cumulative_{trade_type.lower()}'], 
                label=trade_type,
                color=color
            )
    
    axes[1, 1].set_title('Cumulative Number of Trades by Type Over Time', fontsize=14)
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Number of Trades')
    axes[1, 1].legend()
    
    # 6. Win rate by time of day
    hourly_win_rate = combined_df.pivot_table(
        values='is_win' if 'is_win' in combined_df.columns else combined_df['profit_loss'] > 0,
        index='hour',
        columns='trade_type',
        aggfunc='mean'
    ).fillna(0)
    
    sns.heatmap(hourly_win_rate, cmap='RdYlGn', vmin=0, vmax=1, annot=True, fmt=".2f", ax=axes[1, 2])
    axes[1, 2].set_title('Win Rate by Hour of Day', fontsize=14)
    axes[1, 2].set_xlabel('Trade Type')
    axes[1, 2].set_ylabel('Hour of Day')
    
    plt.tight_layout()
    plt.show()
    


    
    # Additional analysis: Trade clustering
    print("\n=== TRADE CLUSTERING ANALYSIS ===")
    print("Analyzing periods with high trade frequency...")
    
    # Group by date and count trades
    date_counts = combined_df.groupby([combined_df['entry_date'].dt.date, 'trade_type']).size().unstack(fill_value=0)
    
    # Find days with highest trading activity
    high_activity_days = date_counts.sum(axis=1).nlargest(5)
    print(f"\nTop 5 days with highest trading activity:")
    for date, count in high_activity_days.items():
        date_df = combined_df[combined_df['entry_date'].dt.date == date]
        win_rate = (date_df['profit_loss'] > 0).mean() * 100
        avg_profit = date_df['profit_loss'].mean()
        print(f"  {date}: {count} trades, Win rate: {win_rate:.1f}%, Avg P&L: ${avg_profit:.2f}")
    
    # Performance by volatility (using ATR as proxy if available)
    if 'ATR_10' in combined_df.columns:
        print("\nPerformance by volatility level (ATR):")
        combined_df['atr_quantile'] = pd.qcut(combined_df['ATR_10'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        atr_perf = combined_df.groupby(['atr_quantile', 'trade_type']).agg({
            'profit_loss': ['mean', 'count'],
            'is_win' if 'is_win' in combined_df.columns else combined_df['profit_loss'] > 0: 'mean'
        })
        print(atr_perf)
    
    return combined_df

# %%
# Call the function with our trade results
combined_df = analyze_trading_seasonality_patterns(long_trade_results, short_trade_results)

# %%
combined_df

# %% [markdown]
# ### Optimization: Finding Better Set of Parameters

# %%
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from itertools import product
import time

def optimize_trading_parameters(df, parameter_grid=None, risk_per_trade_pct=1.0, 
                                             commission_pct=0.075, slippage_pct=0.05, 
                                             trade_type='long', verbose=True):
    """
    Optimize trading parameters by grid search through different parameter combinations
    with progress bar visualization
    
    Parameters:
    - df: DataFrame containing price data
    - parameter_grid: Dictionary with lists of values for each parameter to test
                     (If None, uses default grid)
    - risk_per_trade_pct: Risk percentage per trade
    - commission_pct: Commission percentage
    - slippage_pct: Slippage percentage
    - trade_type: 'long' or 'short' to optimize for specific trade direction
    - verbose: Whether to print progress updates
    
    Returns:
    - DataFrame containing results of all parameter combinations sorted by performance
    - Dictionary with best parameters found
    """
    
    # Default parameter grid if none provided
    if parameter_grid is None:
        parameter_grid = {
            'ema_period': [5, 10, 20, 50],
            'atr_period': [10, 14, 20],
            'target_atr_multiplier': [0.5, 0.75, 1.0, 1.5, 2.0],
            'stop_loss_atr_multiplier': [0.5, 0.75, 1.0, 1.5, 2.0]
        }
    
    # Create all combinations of parameters
    param_keys = list(parameter_grid.keys())
    param_values = list(parameter_grid.values())
    combinations = list(product(*param_values))
    
    if verbose:
        print(f"Testing {len(combinations)} parameter combinations for {trade_type} trades")
        
    # Initialize results storage
    results = []
    start_time = time.time()
    
    # Loop through all parameter combinations with progress bar
    for combo in tqdm(combinations, desc="Optimizing parameters", leave=True):
        # Create parameter dictionary for this combination
        params = dict(zip(param_keys, combo))
            
        # Find trade setups with current parameters
        if trade_type.lower() == 'long':
            _, trade_setups = find_long_trades(
                df, 
                ema_period=params['ema_period'], 
                atr_period=params['atr_period'],
                target_atr_multiplier=params['target_atr_multiplier'],
                stop_loss_atr_multiplier=params['stop_loss_atr_multiplier']
            )
            
            # Skip if no trades found
            if len(trade_setups) == 0:
                continue
                
            # Run backtest with current parameters
            backtest_results, metrics, _ = backtest_long_trade_strategy(
                df, 
                trade_setups,
                risk_per_trade_pct=risk_per_trade_pct,
                commission_pct=commission_pct,
                slippage_pct=slippage_pct,
                target_atr_multiplier=params['target_atr_multiplier'],
                stop_loss_atr_multiplier=params['stop_loss_atr_multiplier']
            )
            
        elif trade_type.lower() == 'short':
            _, trade_setups = find_short_trades(
                df, 
                ema_period=params['ema_period'], 
                atr_period=params['atr_period'],
                target_atr_multiplier=params['target_atr_multiplier'],
                stop_loss_atr_multiplier=params['stop_loss_atr_multiplier']
            )
            
            # Skip if no trades found
            if len(trade_setups) == 0:
                continue
                
            # Run backtest with current parameters
            backtest_results, metrics, _ = backtest_short_trade_strategy(
                df, 
                trade_setups,
                risk_per_trade_pct=risk_per_trade_pct,
                commission_pct=commission_pct,
                slippage_pct=slippage_pct,
                target_atr_multiplier=params['target_atr_multiplier'],
                stop_loss_atr_multiplier=params['stop_loss_atr_multiplier']
            )
        else:
            raise ValueError("trade_type must be 'long' or 'short'")
        
        # Calculate performance scores
        risk_adjusted_return = metrics['total_return_pct'] / metrics['max_drawdown_pct'] if metrics['max_drawdown_pct'] > 0 else 0
        
        # Store results with all metrics and parameters
        result = {
            **params,  # Include all parameters
            'total_trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'total_return_pct': metrics['total_return_pct'],
            'max_drawdown_pct': metrics['max_drawdown_pct'],
            'risk_adjusted_return': risk_adjusted_return,
            'avg_win': metrics['average_win'],
            'avg_loss': metrics['average_loss'],
            'expectancy': metrics['expectancy'],
            'final_balance': metrics['final_balance']
        }
        
        results.append(result)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No valid parameter combinations found")
        return pd.DataFrame(), {}
    
    # Sort results by risk-adjusted return (descending)
    results_df.sort_values('risk_adjusted_return', ascending=False, inplace=True)
    
    # Get best parameters
    best_params = results_df.iloc[0].to_dict()
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    if verbose:
        print("\n==== OPTIMIZATION RESULTS ====")
        print(f"Total combinations tested: {len(combinations)}")
        print(f"Valid combinations found: {len(results_df)}")
        print(f"Best risk-adjusted return: {best_params['risk_adjusted_return']:.4f}")
        print(f"Total time elapsed: {elapsed_time:.2f} seconds")
        print("\nBest parameters:")
        for param in param_keys:
            print(f"  {param}: {best_params[param]}")
        print("\nTop 5 parameter combinations:")
        print(results_df[param_keys + ['risk_adjusted_return', 'total_return_pct', 
                                     'win_rate', 'max_drawdown_pct']].head(5))
    
    return results_df, best_params

# %% [markdown]
# #### Find Optimum Parameter Combination for Long Trades

# %%
parameter_grid = {
    'ema_period': [5, 10, 20],
    'atr_period': [5, 10, 20],
    'target_atr_multiplier': [0.5, 0.75, 1.0, 1.5, 2.0],
    'stop_loss_atr_multiplier': [0.5, 0.75, 1.0, 1.5, 2.0]
}

long_optimization_df, long_best_params = optimize_trading_parameters(df,
                                                      parameter_grid=parameter_grid, 
                                                      trade_type='long', 
                                                      verbose=True)

# %%
long_optimization_df

# %%
long_best_params

# %% [markdown]
# #### Find Optimum Parameter Combination for Short Trades

# %%
parameter_grid = {
    'ema_period': [5, 10, 20],
    'atr_period': [5, 10, 20],
    'target_atr_multiplier': [0.5, 0.75, 1.0, 1.5, 2.0],
    'stop_loss_atr_multiplier': [0.5, 0.75, 1.0, 1.5, 2.0]
}

short_optimization_df, short_best_params = optimize_trading_parameters(df,
                                                      parameter_grid=parameter_grid, 
                                                      trade_type='short', 
                                                      verbose=True)

# %%
short_optimization_df

# %%
short_best_params

# %% [markdown]
# ### Optimization Method for Trading Strategy Parameters
# 
# The optimization method applied in this cryptocurrency trading strategy involves a systematic grid search approach to identify the optimal combination of parameters that produce the best risk-adjusted returns. This comprehensive process enables the discovery of parameter values that maximize profitability while managing risk effectively.
# 
# The optimization process specifically targets four key parameters:
# 
# - **EMA Period**: Periods used for the Exponential Moving Average calculation (5, 10)
# - **ATR Period**: Periods used for the Average True Range calculation (10)
# - **Target ATR Multiplier**: Factor applied to ATR for determining take-profit levels (0.5, 0.75)
# - **Stop-Loss ATR Multiplier**: Factor applied to ATR for determining stop-loss levels (0.75, 1.0)
# 
# Rather than relying on random parameter selection, the optimization systematically evaluates all possible combinations from the predefined parameter grid. This ensures comprehensive coverage of the parameter space.
# 
# For each parameter combination, the algorithm:
# 
# 1. Identifies potential trade setups based on the specified criteria
# 2. Simulates trade execution with realistic conditions including slippage (0.05%) and commission (0.075%)
# 3. Records detailed trade outcomes including profit/loss, win rate, and drawdowns
# 4. Calculates performance metrics for comparative analysis
# 
# The results are ranked primarily by risk-adjusted return, calculated as the ratio between total return percentage and maximum drawdown percentage. This metric provides a balanced view of performance, favoring strategies that generate returns with minimal drawdowns.
# 
# 

# %% [markdown]
# Some features of my analysis:
# 1. **Separate Long & Short Analysis**: Trades are evaluated independently by direction to identify directional biases.
# 2. **ATR-Based Risk Management**: Uses Average True Range (ATR) to set dynamic targets and stop losses based on market volatility.
# 3. **Time-Based Pattern Analysis**: Examines performance by hour of day, day of week, and month to identify temporal patterns.
# 4. **Trade Clustering Analysis**: Identifies periods of high trading activity and their impact on performance.
# 5. **Parameter Optimization**: Implements grid search for finding optimal parameter combinations.
# 

# %% [markdown]
# Ilyas Ustun  
# Chicago, IL  
# 6/15/2025

# %% [markdown]
# 


