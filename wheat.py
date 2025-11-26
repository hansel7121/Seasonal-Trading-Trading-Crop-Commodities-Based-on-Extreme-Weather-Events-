import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    "crops_data/kansas_wheat_temps_10y.csv", index_col="Date", parse_dates=True
)

wheat_prices = yf.download(
    "KE=F", start="2015-01-01", end="2025-11-24", auto_adjust=True
)
if isinstance(wheat_prices.columns, pd.MultiIndex):
    wheat_prices.columns = wheat_prices.columns.droplevel(1)


def plot_temperature(df):
    plt.plot(df.index, (df["Max_Temp_C"] + df["Min_Temp_C"]) / 2)
    plt.title("Average Temperature in Kansas")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.show()


def plot_extremes(df):
    extreme_hots = []
    extreme_colds = []
    plt.plot(df.index, df["Max_Temp_C"], label="Max Temp")
    plt.plot(df.index, df["Min_Temp_C"], label="Min Temp")
    for i in range(len(df)):
        if df["Max_Temp_C"][i] > 35 and df.index[i].month in [5, 6]:
            plt.plot(df.index[i], df["Max_Temp_C"][i], "r^", markersize=10)
            extreme_hots.append(df.index[i].date())
        if df["Min_Temp_C"][i] < -3 and df.index[i].month in [4, 5]:
            plt.plot(df.index[i], df["Min_Temp_C"][i], "b^", markersize=10)
            extreme_colds.append(df.index[i].date())
    plt.title("Extreme Temperatures During wheat Harvest")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.show()
    return extreme_hots, extreme_colds


def plot_prices(prices, extreme_hots, extreme_colds):
    extreme_hots = pd.to_datetime(extreme_hots)
    extreme_colds = pd.to_datetime(extreme_colds)
    plt.plot(prices.index, prices["Close"])
    for date in extreme_hots:
        try:
            plt.plot(date, prices.loc[date]["Close"], "ro", markersize=10)
        except KeyError:
            continue
    for date in extreme_colds:
        try:
            plt.plot(date, prices.loc[date]["Close"], "bo", markersize=10)
        except KeyError:
            continue
    plt.title("wheat Prices During Extreme Temperatures")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.show()


def buy_signals(extremes_hots, extremes_colds, prices):
    all_dates = pd.to_datetime(extremes_hots + extremes_colds)
    all_dates = all_dates.sort_values()
    buy_signals = []
    seen_months = set()

    for date in all_dates:
        if date not in prices.index or date.year == 2025:
            continue
        month_key = (date.year, date.month)
        if month_key in seen_months:
            continue
        seen_months.add(month_key)
        buy_signals.append(date)

    plt.plot(prices.index, prices["Close"])
    for date in buy_signals:
        plt.plot(date, prices.loc[date]["Close"], "go", markersize=10)
    plt.title("Buy Signals")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.show()
    return buy_signals


def plot_returns(prices, buy_signals, holding_period):
    cash = 10000
    portfolio_value = pd.Series(index=prices.index, data=cash, dtype=float)
    buy_signals = sorted(buy_signals)
    busy_until_date = None

    for buy_date in buy_signals:
        if buy_date not in prices.index:
            continue

        if busy_until_date is not None and buy_date < busy_until_date:
            continue

        buy_price = prices.loc[buy_date]["Close"]
        wheat_shares = cash / buy_price
        target_sell_date = buy_date + pd.DateOffset(months=holding_period)
        idx = prices.index.get_indexer([target_sell_date], method="nearest")[0]
        sell_date = prices.index[idx]

        if sell_date > prices.index[-1]:
            break

        period_prices = prices.loc[buy_date:sell_date]["Close"]
        portfolio_value.loc[buy_date:sell_date] = wheat_shares * period_prices
        sell_price = prices.loc[sell_date]["Close"]
        cash = wheat_shares * sell_price
        portfolio_value.loc[sell_date:] = cash
        busy_until_date = sell_date

    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_value.index, portfolio_value, label="Portfolio Value")
    plt.title("Portfolio Value Over Time (Initial Cash: $10,000)")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.grid(True)
    plt.legend()
    plt.show()

    total_return = (cash - 10000) / 10000
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1
    print(f"Final Portfolio Value: ${cash:.2f}")
    print(f"Annualized Return: {annualized_return * 100:.2f}%")

    return cash, annualized_return


extreme_hots, extreme_colds = plot_extremes(df)
print(extreme_hots)
print(extreme_colds)
plot_prices(wheat_prices, extreme_hots, extreme_colds)
buy_signals = buy_signals(extreme_hots, extreme_colds, wheat_prices)
cash, annual_returns = plot_returns(wheat_prices, buy_signals, 6)

# Example of winter wheat not working because it is really resistant to changes to temperature.
