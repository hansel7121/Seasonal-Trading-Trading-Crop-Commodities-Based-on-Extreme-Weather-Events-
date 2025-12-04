import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

coffee_df = pd.read_csv(
    "crops_data/varginha_coffee_temps_10y.csv", index_col="Date", parse_dates=True
)

coffee_prices = yf.download(
    "KC=F", start="2015-01-01", end="2025-11-24", auto_adjust=True
)
if isinstance(coffee_prices.columns, pd.MultiIndex):
    coffee_prices.columns = coffee_prices.columns.droplevel(1)


def plot_extremes(df):
    extreme_hots = []
    extreme_colds = []
    hot_labeled = False
    cold_labeled = False
    plt.plot(df.index, df["Max_Temp_C"], label="Max Temp")
    plt.plot(df.index, df["Min_Temp_C"], label="Min Temp")
    for i in range(len(df)):
        if df["Max_Temp_C"].iloc[i] > 33 and df.index[i].month in [9, 10]:
            if not hot_labeled:
                plt.plot(
                    df.index[i],
                    df["Max_Temp_C"].iloc[i],
                    "r^",
                    markersize=10,
                    label="Extreme Hot",
                )
                hot_labeled = True
            else:
                plt.plot(df.index[i], df["Max_Temp_C"].iloc[i], "r^", markersize=10)
            extreme_hots.append(df.index[i].date())
        if df["Min_Temp_C"].iloc[i] < 2 and df.index[i].month in [6, 7, 8]:
            if not cold_labeled:
                plt.plot(
                    df.index[i],
                    df["Min_Temp_C"].iloc[i],
                    "b^",
                    markersize=10,
                    label="Extreme Cold",
                )
                cold_labeled = True
            else:
                plt.plot(df.index[i], df["Min_Temp_C"].iloc[i], "b^", markersize=10)
            extreme_colds.append(df.index[i].date())
    plt.title("Extreme Temperatures During coffee Harvest")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.show()
    return pd.to_datetime(extreme_hots), pd.to_datetime(extreme_colds)


def plot_prices(prices, extreme_hots, extreme_colds):
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
    plt.title("coffee Prices During Extreme Temperatures")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.show()


def buy_signals(extremes_hots, extremes_colds, prices):
    all_dates = extremes_hots.union(extremes_colds)
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


def backtest_strategy(prices, buy_signals, holding_period):
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
        coffee_shares = cash / buy_price
        target_sell_date = buy_date + pd.DateOffset(months=holding_period)
        idx = prices.index.get_indexer([target_sell_date], method="nearest")[0]
        sell_date = prices.index[idx]

        if sell_date > prices.index[-1]:
            break

        period_prices = prices.loc[buy_date:sell_date]["Close"]
        portfolio_value.loc[buy_date:sell_date] = coffee_shares * period_prices
        sell_price = prices.loc[sell_date]["Close"]
        cash = coffee_shares * sell_price
        portfolio_value.loc[sell_date:] = cash
        busy_until_date = sell_date

    total_return = (cash - 10000) / 10000
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1
    print(f"Final Portfolio Value: ${cash:.2f}")
    print(f"Annualized Return: {annualized_return * 100:.2f}%")

    return cash, annualized_return, portfolio_value


def plot_returns(prices, buy_signals, holding_period):
    cash, annualized_return, portfolio_value = backtest_strategy(
        prices, buy_signals, holding_period
    )
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_value.index, portfolio_value, label="Portfolio Value")
    plt.title(f"Portfolio Value Over {holding_period} Months (Initial Cash: $10,000)")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.grid(True)
    plt.legend()
    plt.show()


def get_coffee_buy_signals():
    extreme_hots = []
    extreme_colds = []
    for i in range(len(coffee_df)):
        if coffee_df["Max_Temp_C"].iloc[i] > 33 and coffee_df.index[i].month in [9, 10]:
            extreme_hots.append(coffee_df.index[i].date())
        if coffee_df["Min_Temp_C"].iloc[i] < 2 and coffee_df.index[i].month in [6, 7, 8]:
            extreme_colds.append(coffee_df.index[i].date())

    extreme_hots = pd.to_datetime(extreme_hots)
    extreme_colds = pd.to_datetime(extreme_colds)

    all_dates = extreme_hots.union(extreme_colds)
    all_dates = all_dates.sort_values()
    signals = []
    seen_months = set()

    for date in all_dates:
        if date not in coffee_prices.index or date.year == 2025:
            continue
        month_key = (date.year, date.month)
        if month_key in seen_months:
            continue
        seen_months.add(month_key)
        signals.append(date)

    return signals


def optimize_holding_period(prices, buy_signals, min_months=1, max_months=12):
    print(f"--- Optimizing Strategy ({min_months}-{max_months} months) ---")

    cash_results = {}
    return_results = {}
    best_cash = 0
    best_month = 0

    for m in range(min_months, max_months + 1):
        cash, annualized_return, portfolio_value = backtest_strategy(
            prices, buy_signals, m
        )
        profit = cash - 10000
        print(
            f"Holding: {m} months | Final Cash: ${cash:,.2f} | Profit: ${profit:,.2f}"
        )
        cash_results[m] = cash
        return_results[m] = profit / 10000
        if cash > best_cash:
            best_cash = cash
            best_month = m

    return best_month, best_cash, cash_results, return_results


def plot_optimization_results(cash_results, return_results, best_months):
    cash_periods = list(cash_results.keys())
    cash_values = list(cash_results.values())
    return_values = list(return_results.values())

    fig, ax1 = plt.subplots(figsize=(12, 6))

    bars = ax1.bar(
        cash_periods, cash_values, color="skyblue", alpha=0.7, label="Portfolio Value"
    )
    bars[best_months - 1].set_color("green")
    ax1.axhline(
        y=10000,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Starting Cash ($10k)",
    )
    ax1.set_xlabel("Holding Period (Months)")
    ax1.set_ylabel("Final Portfolio Value ($)")
    ax1.tick_params(axis="y")
    ax1.set_xticks(cash_periods)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    line = ax2.plot(
        cash_periods,
        return_values,
        color="darkgreen",
        marker="o",
        linewidth=2,
        markersize=6,
        label="% Return",
    )
    ax2.set_ylabel("Percentage Return (%)")
    ax2.tick_params(axis="y", labelcolor="darkgreen")
    ax2.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    plt.title("Strategy Performance by Holding Period")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.show()


coffee_buy_signals = None

if __name__ == "__main__":
    extreme_hots, extreme_colds = plot_extremes(coffee_df)
    plot_prices(coffee_prices, extreme_hots, extreme_colds)
    coffee_buy_signals = buy_signals(extreme_hots, extreme_colds, coffee_prices)
    print(coffee_buy_signals)
    cash, annualized_return, portfolio_value = backtest_strategy(
        coffee_prices, coffee_buy_signals, 6
    )
    plot_returns(coffee_prices, coffee_buy_signals, 6)
    best_months, best_pnl, cash_results, return_results = optimize_holding_period(
        coffee_prices, coffee_buy_signals, 1, 12
    )
    plot_optimization_results(cash_results, return_results, best_months)