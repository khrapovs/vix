import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd


def import_yields() -> pd.DataFrame:
    """Import yields.

    Date,Days,Rate
    20090101,9,0.38
    20090101,37,0.38

    """
    yields = pd.read_csv(
        Path(__file__).parents[2] / "data" / "yields.csv",
        converters={"Date": lambda x: dt.datetime.strptime(x, "%Y%m%d")},
    )
    return yields.set_index(["Date", "Days"])


def import_options() -> pd.DataFrame:
    """Import options.

    Expiration,Days,Strike,Call Bid,Call Ask,Put Bid,Put Ask
    20090110,9,200,717.6,722.8,0,0.05
    20090110,9,250,667.6,672.9,0,0.05
    20090110,9,300,617.9,622.9,0,0.05
    20090110,9,350,567.9,572.9,0,0.05
    20090110,9,375,542.9,547.9,0,0.1

    """
    # Function to parse dates of '20090101' format
    raw_options = pd.read_csv(
        Path(__file__).parents[2] / "data" / "options.csv",
        converters={"Expiration": lambda x: dt.datetime.strptime(x, "%Y%m%d")},
    )

    # Function to convert days to internal timedelta format
    raw_options["Date"] = raw_options["Expiration"] - raw_options["Days"].map(lambda x: dt.timedelta(days=int(x)))
    # Convert integer strikes to float!
    # Otherwise it may lead to accumulation of errors.
    raw_options["Strike"] = raw_options["Strike"].astype(float)

    return raw_options


def clean_options(options: pd.DataFrame) -> pd.DataFrame:
    """Clean and index option data set."""
    # Since VIX is computed for the date of option quotations,
    # we do not really need Expiration
    options.set_index(["Date", "Days", "Strike"], inplace=True)
    options.drop("Expiration", axis=1, inplace=True)

    # Do some renaming and separate calls from puts
    calls = options[["Call Bid", "Call Ask"]].rename(columns={"Call Bid": "Bid", "Call Ask": "Ask"})
    puts = options[["Put Bid", "Put Ask"]].rename(columns={"Put Bid": "Bid", "Put Ask": "Ask"})

    # Add a column indicating the type of the option
    calls["CP"], puts["CP"] = "C", "P"

    # Merge calls and puts
    options = pd.concat([calls, puts])

    # Reindex and sort
    options.reset_index(inplace=True)
    options.set_index(["Date", "Days", "CP", "Strike"], inplace=True)
    options.sort_index(inplace=True)

    return options


def bid_ask_average(options: pd.DataFrame) -> pd.DataFrame:
    """Compute bid/ask average."""
    # This step is used further to filter out in-the-money options.

    options["Premium"] = (options["Bid"] + options["Ask"]) / 2
    options = options[options["Bid"] > 0]["Premium"].unstack("CP")
    return options


def put_call_parity(options: pd.DataFrame) -> pd.DataFrame:
    """Find put-call parity."""
    # Find the absolute difference
    options["CPdiff"] = (options["C"] - options["P"]).abs()
    # Mark the minimum for each date/term
    grouped = options["CPdiff"].groupby(level=["Date", "Days"])
    options["min"] = grouped.transform(lambda x: x == x.min())
    return options


def forward_price(options: pd.DataFrame, yields: pd.DataFrame) -> pd.DataFrame:
    """Compute forward price."""
    # Leave only at-the-money options
    df = options[options["min"] == 1].reset_index("Strike")
    df = df.groupby(level=["Date", "Days"]).first().reset_index()
    # Merge with risk-free rate
    df = pd.merge(df, yields.reset_index(), how="left")

    # Compute the implied forward
    df["Forward"] = df["CPdiff"] * np.exp(df["Rate"] * df["Days"] / 36500)
    df["Forward"] += df["Strike"]
    forward = df.set_index(["Date", "Days"])[["Forward"]]
    return forward


def at_the_money_strike(options: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
    """Compute at-the-money strike."""
    # Merge options with implied forward price
    left = options.reset_index().set_index(["Date", "Days"])
    df = pd.merge(left, forward, left_index=True, right_index=True)
    # Compute at-the-money strike
    df = df[df["Strike"] < df["Forward"]]["Strike"]
    mid_strike = df.groupby(level=["Date", "Days"]).max()
    mid_strike = pd.DataFrame({"Mid Strike": mid_strike})
    return mid_strike


def leave_out_of_the_money(options: pd.DataFrame, mid_strike: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate out-of-the-money calls and puts."""
    # Go back to original data and reindex it
    left = options.reset_index()
    left.set_index(["Date", "Days"], inplace=True)
    left.drop("Premium", axis=1, inplace=True)
    # Merge with at-the-money strike
    df = pd.merge(left, mid_strike, left_index=True, right_index=True)
    # Separate out-of-the-money calls and puts
    P = (df["Strike"] <= df["Mid Strike"]) & (df["CP"] == "P")
    C = (df["Strike"] >= df["Mid Strike"]) & (df["CP"] == "C")
    puts, calls = df[P], df[C]
    return puts, calls


def remove_crazy_quotes(calls: pd.DataFrame, puts: pd.DataFrame) -> pd.DataFrame:
    """Remove all quotes after two consequtive zero bids."""
    # Indicator of zero bid
    calls["zero_bid"] = (calls["Bid"] == 0).astype(int)
    # Accumulate number of zero bids starting at-the-money
    grouped = calls.groupby(level=["Date", "Days"])["zero_bid"]
    calls["zero_bid_accum"] = grouped.cumsum()

    # Sort puts in reverse order inside date/term
    grouped = puts.groupby(level=["Date", "Days"])
    puts = grouped.apply(lambda x: x.sort_values(by="Strike", ascending=False))
    # Indicator of zero bid
    puts["zero_bid"] = (puts["Bid"] == 0).astype(int)
    # Accumulate number of zero bids starting at-the-money
    grouped = puts.groupby(level=["Date", "Days"])["zero_bid"]
    puts["zero_bid_accum"] = grouped.cumsum()

    # Merge puts and calls
    options3 = pd.concat([calls, puts]).reset_index()
    # Throw away bad stuff
    options3 = options3[(options3["zero_bid_accum"] < 2) & (options3["Bid"] > 0)]
    # Compute option premium as bid/ask average
    options3["Premium"] = (options3["Bid"] + options3["Ask"]) / 2
    options3.set_index(["Date", "Days", "CP", "Strike"], inplace=True)
    options3 = options3["Premium"].unstack("CP")

    return options3


def out_of_the_money_options(options3: pd.DataFrame, mid_strike: pd.DataFrame) -> pd.DataFrame:
    """Compute out-of-the-money option price."""
    # Merge wth at-the-money strike price
    left = options3.reset_index().set_index(["Date", "Days"])
    df = pd.merge(left, mid_strike, left_index=True, right_index=True)

    # Conditions to separate out-of-the-money puts and calls
    condition1 = df["Strike"] < df["Mid Strike"]
    condition2 = df["Strike"] > df["Mid Strike"]
    # At-the-money we have two quotes, so take the average
    df["Premium"] = (df["P"] + df["C"]) / 2
    # Remove in-the-money options
    df.loc[condition1, "Premium"] = df.loc[condition1, "P"]
    df.loc[condition2, "Premium"] = df.loc[condition2, "C"]

    options4 = df[["Strike", "Mid Strike", "Premium"]].copy()
    return options4


def compute_adjoining_strikes_diff(group: pd.DataFrame) -> pd.DataFrame:
    new = group.copy()
    new.iloc[1:-1] = np.array((group.values[2:] - group.values[:-2]) / 2)
    new.iloc[0] = group.values[1] - group.values[0]
    new.iloc[-1] = group.values[-1] - group.values[-2]
    return new


def strike_diff(options: pd.DataFrame) -> pd.DataFrame:
    """Compute difference between adjoining strikes."""
    options["dK"] = options.groupby(level=["Date", "Days"])["Strike"].transform(compute_adjoining_strikes_diff)
    return options


def strike_contribution(options4: pd.DataFrame, yields: pd.DataFrame) -> pd.DataFrame:
    """Compute contribution of each strike."""
    # Merge with risk-free rate
    contrib = pd.merge(options4, yields, left_index=True, right_index=True)
    contrib.reset_index(inplace=True)

    contrib["sigma2"] = contrib["dK"] / contrib["Strike"] ** 2
    contrib["sigma2"] *= contrib["Premium"] * np.exp(contrib["Rate"] * contrib["Days"] / 36500)

    return contrib


def each_period_vol(contrib: pd.DataFrame, mid_strike: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
    """Compute each preiod index."""
    # Sum up contributions from all strikes
    sigma2 = contrib.groupby(["Date", "Days"])[["sigma2"]].sum() * 2

    # Merge at-the-money strike and implied forward
    sigma2["Mid Strike"] = mid_strike
    sigma2["Forward"] = forward

    # Compute variance for each term
    sigma2["sigma2"] -= (sigma2["Forward"] / sigma2["Mid Strike"] - 1) ** 2
    sigma2["sigma2"] /= sigma2.index.get_level_values(1).astype(float) / 365
    sigma2 = sigma2[["sigma2"]]

    return sigma2


def near_next_term(group: pd.DataFrame) -> pd.DataFrame:
    """Determine near- and next-term if there are several maturities in the data."""
    days = np.array(group["Days"])
    sigma2 = np.array(group["sigma2"])

    if days.min() < 30:
        T1 = days[days < 30].max()
        T2 = days[days > T1].min()
    elif (days.min() == 30) or (days.max() == 30):
        T1 = T2 = 30
    else:
        T1 = days.min()
        T2 = days[days > T1].min()

    sigma_T1 = sigma2[days == T1][0]
    sigma_T2 = sigma2[days == T2][0]
    data = [{"T1": T1, "T2": T2, "sigma2_T1": sigma_T1, "sigma2_T2": sigma_T2}]

    return pd.DataFrame(data)


def interpolate_vol(sigma2: pd.DataFrame) -> pd.DataFrame:
    """Compute interpolated index."""
    grouped = sigma2.reset_index().groupby("Date")
    two_sigmas = grouped.apply(near_next_term, include_groups=False).groupby(level="Date").first()
    return two_sigmas


def interpolate_vix(two_sigmas: pd.DataFrame) -> pd.DataFrame:
    """Interpolate the VIX."""
    df = two_sigmas.copy()

    for t in ["T1", "T2"]:
        # Convert to fraction of the year
        df["days_" + t] = df[t].astype(float) / 365
        # Convert to miutes
        df[t] = (df[t] - 1) * 1440 + 510 + 930

    denom = df["T2"] - df["T1"]
    if denom.iloc[0] > 0:
        coef1 = df["days_T1"] * (df["T2"] - 30 * 1440) / denom
        coef2 = df["days_T2"] * (30 * 1440 - df["T1"]) / denom
    else:
        coef1 = coef2 = df["days_T1"]
    df["sigma2_T1"] = df["sigma2_T1"] * coef1
    df["sigma2_T2"] = df["sigma2_T2"] * coef2
    df["VIX"] = ((df["sigma2_T1"] + df["sigma2_T2"]) * 365 / 30) ** 0.5 * 100

    return df


def whitepaper() -> pd.DataFrame:
    ### Import yields
    yields = import_yields()

    ### Import options
    raw_options = import_options()

    # Uncomment this block to check that VIX is computed,
    # when there are options with exactly 30 days to expire.
    #    yields.reset_index('Days', inplace=True)
    #    yields['Days'] += 21
    #    yields.set_index('Days', inplace=True, append=True)
    #    raw_options['Days'] += 21

    ### Do some cleaning and indexing
    options = clean_options(raw_options)

    ### Compute bid/ask average
    options2 = bid_ask_average(options)

    ### Put-call parity
    options2 = put_call_parity(options2)

    ### Compute forward price
    forward = forward_price(options2, yields)

    ### Compute at-the-money strike
    mid_strike = at_the_money_strike(options2, forward)

    ### Separate out-of-the-money calls and puts
    puts, calls = leave_out_of_the_money(options, mid_strike)

    ### Remove all quotes after two consequtive zero bids
    options3 = remove_crazy_quotes(calls, puts)

    ### Compute out-of-the-money option price
    options4 = out_of_the_money_options(options3, mid_strike)

    ### Compute difference between adjoining strikes
    options4 = strike_diff(options4)

    ### Compute contribution of each strike
    contrib = strike_contribution(options4, yields)

    ### Compute each preiod index
    sigma2 = each_period_vol(contrib, mid_strike, forward)

    ### Compute interpolated index
    two_sigmas = interpolate_vol(sigma2)

    ### Interpolate the VIX
    df = interpolate_vix(two_sigmas)

    return df[["VIX"]]


if __name__ == "__main__":
    vixvalue = whitepaper()
    print(vixvalue)
