import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


# Import corn and coffee prices
corn_prices = yf.download("ZC=F", start="2015-01-01", end="2025-11-24", auto_adjust=True)["Close"]
if isinstance(corn_prices.columns, pd.MultiIndex):
    corn_prices.columns = corn_prices.columns.droplevel(1)

coffee_prices = yf.download("KC=F", start="2015-01-01", end="2025-11-24", auto_adjust=True)["Close"]
if isinstance(coffee_prices.columns, pd.MultiIndex):
    coffee_prices.columns = coffee_prices.columns.droplevel(1)

hogs_prices = yf.download("HE=F", start="2015-01-01", end="2025-11-24", auto_adjust=True)["Close"]
if isinstance(hogs_prices.columns, pd.MultiIndex):
    hogs_prices.columns = hogs_prices.columns.droplevel(1)

corn_name = "corn"
coffee_name = "coffee"
hogs_name = "lean hogs"

def combined_prices(prices_series_1, prices_series_2, contract_name_1, contract_name_2):
    combined_prices = pd.concat([prices_series_1, prices_series_2], axis=1)
    combined_prices.columns = [contract_name_1, contract_name_2]
    return combined_prices

combined_prices_df = combined_prices(corn_prices, hogs_prices, corn_name, hogs_name)
print(combined_prices_df.head())


# Import buy signals
from corn.corn_roll_yield import get_corn_buy_signals
from coffee.coffee_roll_yield import get_coffee_buy_signals
from lean_hogs.lean_hogs_roll_yield import get_hogs_buy_signals

corn_buy_signals = sorted(get_corn_buy_signals())
coffee_buy_signals = sorted(get_coffee_buy_signals())
hogs_buy_signals = sorted(get_hogs_buy_signals())


def combined_buy_signals(buy_signals_1, buy_signals_2, contract_name_1, contract_name_2):
    df_1 = pd.DataFrame({'commodity type': contract_name_1, 'date': buy_signals_1})
    df_2 = pd.DataFrame({'commodity type': contract_name_2, 'date': buy_signals_2})
    combined_df = pd.concat([df_1, df_2], ignore_index=True)
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    result_df = combined_df.sort_values(by='date').reset_index(drop=True)
    return result_df

combined_buy_signals_df = combined_buy_signals(corn_buy_signals, hogs_buy_signals, corn_name, hogs_name)
print(combined_buy_signals_df.head())


# Calculate estimated drag
corn_holding_period = 10
coffee_holding_period = 7
hogs_holding_period = 6

corn_estimated_drag = 0.02
coffee_estimated_drag = 0.015
hogs_estimated_drag = 0.025

def get_roll_months(current_date, contract_type):
    month = current_date.month
    if contract_type == "hogs":
        if month in [2, 4, 6, 8, 10, 12]:
            return True
    elif contract_type == "soybeans":
        if month in [1, 3, 5, 7, 8, 9, 11]:
            return True
    else:
        if month in [3, 5, 7, 9, 12]:
            return True
    return False

def get_estimated_drag(buy_date, contract_drag, holding_period, contract_type):
    roll_months = []
    for i in range(1, holding_period + 1):
        if get_roll_months(buy_date + pd.DateOffset(months=i), contract_type):
            roll_months.append(i)
    total_drag = 1
    for i in range(len(roll_months)):
        total_drag *= 1 - contract_drag
    return total_drag


# Backtesting
def portfolio_backtest(prices_df, buy_signals_df, contract_1_name, contract_2_name, contract_1_holding_period, contract_2_holding_period, contract_1_drag, contract_2_drag):
    initial_cash = 10000
    cash_series = pd.Series(index=prices_df.index, data=initial_cash, dtype=float)
    portfolio_1_value = pd.Series(index=prices_df.index, data=0, dtype=float)
    portfolio_2_value = pd.Series(index=prices_df.index, data=0, dtype=float)
    portfolio_1_busy_until_date = None
    portfolio_2_busy_until_date = None
    portfolio_1_cash = 0
    portfolio_2_cash = 0
    portfolio_1_shares = 0
    portfolio_2_shares = 0

    for i in range(len(buy_signals_df)):
        buy_date = buy_signals_df['date'].iloc[i]
        buy_type = buy_signals_df['commodity type'].iloc[i]

        if buy_date not in prices_df.index:
            continue

        holding_contract_1 = portfolio_1_busy_until_date is not None and buy_date < portfolio_1_busy_until_date
        holding_contract_2 = portfolio_2_busy_until_date is not None and buy_date < portfolio_2_busy_until_date


        if buy_type == contract_1_name:
            if holding_contract_1:
                continue

            # buy process
            total_drag = get_estimated_drag(buy_date, contract_1_drag, contract_1_holding_period, contract_1_name)
            buy_price = prices_df.loc[buy_date, contract_1_name]

            current_cash = cash_series.loc[buy_date]

            if holding_contract_2:
                trade_cash = current_cash
            else:
                trade_cash = current_cash / 2

            portfolio_1_shares = trade_cash / buy_price

            target_sell_date = buy_date + pd.DateOffset(months=contract_1_holding_period)
            idx = prices_df.index.get_indexer([target_sell_date], method="nearest")[0]
            sell_date = prices_df.index[idx]

            if sell_date > prices_df.index[-1]:
                continue

            cash_series.loc[buy_date:] -= trade_cash

            portfolio_1_period_prices = prices_df.loc[buy_date:sell_date][contract_1_name]
            portfolio_1_value.loc[buy_date:sell_date] = portfolio_1_shares * portfolio_1_period_prices

            portfolio_1_sell_price = prices_df.loc[sell_date, contract_1_name]
            sell_proceeds = portfolio_1_shares * portfolio_1_sell_price * total_drag

            cash_series.loc[sell_date:] += sell_proceeds

            portfolio_1_value.loc[sell_date:] = 0
            portfolio_1_busy_until_date = sell_date 
        
        if buy_type == contract_2_name:
            if holding_contract_2:
                continue
                
            # buy process
            total_drag = get_estimated_drag(buy_date, contract_2_drag, contract_2_holding_period, contract_2_name)
            buy_price = prices_df.loc[buy_date, contract_2_name]

            current_cash = cash_series.loc[buy_date]

            if holding_contract_1:
                trade_cash = current_cash
            else:
                trade_cash = current_cash / 2

            portfolio_2_shares = trade_cash / buy_price

            target_sell_date = buy_date + pd.DateOffset(months=contract_2_holding_period)
            idx = prices_df.index.get_indexer([target_sell_date], method="nearest")[0]
            sell_date = prices_df.index[idx]

            if sell_date > prices_df.index[-1]:
                continue

            cash_series.loc[buy_date:] -= trade_cash

            portfolio_2_period_prices = prices_df.loc[buy_date:sell_date][contract_2_name]
            portfolio_2_value.loc[buy_date:sell_date] = portfolio_2_shares * portfolio_2_period_prices

            portfolio_2_sell_price = prices_df.loc[sell_date, contract_2_name]
            sell_proceeds = portfolio_2_shares * portfolio_2_sell_price * total_drag

            cash_series.loc[sell_date:] += sell_proceeds

            portfolio_2_value.loc[sell_date:] = 0
            portfolio_2_busy_until_date = sell_date
    

    # Calculate total portfolio value
    portfolio_value = portfolio_1_value + portfolio_2_value + cash_series

    final_portfolio_value = portfolio_value.iloc[-1]
    total_return = (final_portfolio_value - initial_cash) / initial_cash
    years = (prices_df.index[-1] - prices_df.index[0]).days / 365
    annualized_return = (1 + total_return) ** (1 / years) - 1

    print(f"Final Portfolio Value: {final_portfolio_value}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")


    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_value.index, portfolio_value, label="Portfolio Value")
    plt.plot(portfolio_1_value.index, portfolio_1_value, label="Portfolio 1 Value")
    plt.plot(portfolio_2_value.index, portfolio_2_value, label="Portfolio 2 Value")
    plt.title(f"Portfolio Value Over 10 Years (Initial Cash: $10,000)")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
portfolio_backtest(combined_prices_df, combined_buy_signals_df, corn_name, hogs_name, corn_holding_period, hogs_holding_period, corn_estimated_drag, hogs_estimated_drag)
            