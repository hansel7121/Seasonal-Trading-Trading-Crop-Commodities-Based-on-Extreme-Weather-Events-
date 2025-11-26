import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datascience.util import make_array
from datascience import *
from datetime import datetime
from coffee import get_coffee_buy_signals

coffee_df = pd.read_csv(
    "crops_data/varginha_coffee_temps_10y.csv", index_col="Date", parse_dates=True
)
coffee_prices = yf.download(
    "KC=F", start="2015-01-01", end="2025-11-24", auto_adjust=True
)["Close"]
if isinstance(coffee_prices.columns, pd.MultiIndex):
    coffee_prices.columns = coffee_prices.columns.droplevel(1)

coffee_buy_signals = sorted(get_coffee_buy_signals())
print(coffee_buy_signals)

# null hypothesis: positive return is due to random chance
# alternate hypothesis: positive return is due to the strategy

coffee_signals_in_months = make_array()
for signal in coffee_buy_signals:
    period = signal.to_period("M")
    # period = signal.strftime('%Y-%m')
    if period not in coffee_signals_in_months:
        coffee_signals_in_months = np.append(coffee_signals_in_months, period)
#print(coffee_signals_in_months)


# generate array of all months from 2015-01 to 2024-12
first_date = pd.Timestamp(2015, 1, 1)
every_month = make_array()
for i in range(120):
    time_delta = pd.DateOffset(months=i)
    new_month = (first_date + time_delta).to_period("M")
    every_month = np.append(every_month, new_month)
# print(every_month)


# add True to yes_buy_signals_months if that month has a buy signal
yes_buy_signals_months = np.array([], dtype=bool)
for month in every_month:
    if month in coffee_signals_in_months:
        yes_buy_signals_months = np.append(yes_buy_signals_months, True)
    else:
        yes_buy_signals_months = np.append(yes_buy_signals_months, False)
# print(yes_buy_signals_months)


# function to calculate returns for a given month
def month_return(prices, buy_signal, holding_period):
    # Check if buy_signal exists in the index, if not find nearest
    if buy_signal not in prices.index:
        idx = prices.index.get_indexer([buy_signal], method="nearest")[0]
        buy_signal = prices.index[idx]

    buy_price = prices.loc[buy_signal]
    target_sell_date = buy_signal + pd.DateOffset(months=holding_period)

    # Find nearest trading day for sell date
    idx = prices.index.get_indexer([target_sell_date], method="nearest")[0]
    sell_date = prices.index[idx]

    sell_price = prices.loc[sell_date]
    annualized_return = (sell_price - buy_price) / buy_price

    # Convert to float if it's a Series
    if isinstance(annualized_return, pd.Series):
        return float(annualized_return.iloc[0])
    return float(annualized_return)


# calculate returns for every month
ten_month_return_every_month = []

for i in range(len(every_month)):
    annualized_return = month_return(coffee_prices, every_month[i].to_timestamp(), 7)
    ten_month_return_every_month.append(annualized_return)


table = Table().with_columns(
    "Month", every_month,
    "Buy Signal", yes_buy_signals_months,
    "Monthly Return", ten_month_return_every_month,
)
# print(table)


# observed data
buy_signals_with_returns = table.select("Buy Signal", "Monthly Return")
#print(buy_signals_with_returns)
means_table = buy_signals_with_returns.group("Buy Signal", np.average)
#print(means_table)

# Create histogram grouped by Buy Signal
buy_signals_with_returns.hist("Monthly Return", group="Buy Signal")
plt.title("Observed Distribution of Coffee Monthly Return Based on Buy Signals")
plt.savefig('coffee_monthly_returns_histogram.png', dpi=300, bbox_inches='tight')
print("Histogram saved as 'coffee_monthly_returns_histogram.png'")
plt.close()  # Close the figure to free memory

means = means_table.column("Monthly Return average")
observed_difference = means[1] - means[0]
print("Observed difference in means: " + str(observed_difference))


# function to calculate difference in means given table with Buy Signal and Monthly Return columns
def difference_in_means(table, group_label):
    reduced = table.select("Monthly Return", group_label)
    means_table = reduced.group(group_label, np.mean)
    means = means_table.column(1)
    return means[1] - means[0]
# print(difference_in_means(table, "Buy Signal"))


# shuffles the buy signals labels to perform AB testing
def one_simulated_difference():
    shuffled_labels = buy_signals_with_returns.sample(with_replacement=False).column(0)
    shuffled_table = buy_signals_with_returns.select("Monthly Return").with_column(
        "Shuffled Signals", shuffled_labels
    )
    return difference_in_means(shuffled_table, "Shuffled Signals")


# simulate shuffling 5000 times
differences = make_array()
repetition = 5000

for i in np.arange(repetition):
    new_difference = one_simulated_difference()
    differences = np.append(differences, new_difference)


# visualize the results
Table().with_column("Difference Between Group Means", differences).hist()
plt.axvline(observed_difference, color='red', linestyle='--', linewidth=2, label=f'Observed Difference: {observed_difference:.4f}')
plt.legend()
plt.title("Prediction Under the Null Hypothesis for Coffee")
plt.savefig('coffee_null_hypothesis_distribution.png', dpi=300, bbox_inches='tight')
print("Null hypothesis distribution saved as 'coffee_null_hypothesis_distribution.png'")
plt.close()

empirical_p_value = np.count_nonzero(differences >= observed_difference) / repetition
print("Empirical p-value: ", empirical_p_value)
