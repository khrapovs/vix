#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
VIX replication
===============

This file shows how to reproduce the VIX given the data in [1]_. The code
works for any option data set, not only one day as in the White Paper. The
option data for this example is exactly the same as in the Appendix 1 of the
White Paper.

Given are the prices :math:`C_{i}`, :math:`i\in\left\{ 0,\ldots,n\right\}`, of
a series of European call options on the index with fixed maturity date
:math:`T` and exercise prices :math:`K_{i}`, :math:`i\in\left\{
0,\ldots,n\right\}`, as well as the prices :math:`P_{i}`, :math:`i\in\left\{
0,\ldots,n\right\}`, of a series of European put options on the index with the
same maturity date :math:`T` and exercise prices :math:`K_{i}`. Let further
hold :math:`K_{i}<K_{i+1}` for all :math:`i\in\left\{ 0,\ldots,n-1\right\}`.

The VIX itself is

.. math::
    VIX=100\cdot\sqrt{V^{2}},

where :math:`V` is explained below.

Since there are days when there no options with precisely 30 days to
expiration, we have to interpolate between near-term index and next-term
index:

.. math::
    V^{2}=\left[T_{1}\sigma_{1}^{2}\left(\frac{N_{T_{2}}-N_{30}}
    {N_{T_{2}}-N_{T_{1}}}\right)+T_{2}
    \sigma_{2}^{2}\left(\frac{N_{30}-N_{T_{1}}}{N_{T_{2}}
    -N_{T_{1}}}\right)\right]\frac{365}{30}

with each :math:`\sigma_{i}^{2}` computed
according to

.. math::
    \sigma^{2}=\frac{2}{T}\sum_{i=0}^{n}\frac{\Delta
    K_{i}}{K_{i}^{2}}e^{rT}M_{i}-\frac{1}{T}\left(\frac{F}{K_{*}}-1\right)^{2},

where the distance between strikes is

.. math::
    \Delta K_{i}  =   \begin{cases}
    K_{1}-K_{0}, & i=0\\ \frac{1}{2}\left(K_{i+1}-K_{i-1}\right), &
    i=1,\ldots,n-1\\ K_{n}-K_{n-1}, & i=n \end{cases}

the out-of-the-money option premium is

.. math::
    M_{i} =   \begin{cases} P_{i}, & K_{i}<K_{*}\\
    \frac{1}{2}\left(P_{i}+C_{i}\right), & K_{i}=K_{*}\\ C_{i}, & K_{i}>K_{*}
    \end{cases}

at-the-money strike price is

.. math::
    K_{*} =   \max\left\{
    K_{i}<F\right\},

forward price extracted from put-call parity:

.. math::
    F =
    K_{j}+e^{rT}\left|C_{j}-P_{j}\right|,

with

.. math::
    j=\min\left\{
    \left|C_{i}-P_{i}\right|\right\},

and finally :math:`r` is the constant risk-free
short rate appropriate for maturity :math:`T`.

References
----------
.. [1] CBOE White Paper, http://www.cboe.com/micro/vix/vixwhite.pdf

"""
from __future__ import print_function, division

import numpy as np
import pandas as pd
import datetime as dt

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"


def whitepaper():
    ### Import yields

    # In the white paper it is assumed that the risk-free rate is 0.38%
    # for both near- and nex-term options.
    # The date chosen for computation is Jan 1, 2009 with options expiring
    # in 9 and 37 days.

    f = lambda x: dt.datetime.strptime(x, '%Y%m%d')
    yields = pd.read_csv('../data/yields.csv', converters={'Date': f})
    yields = yields.set_index(['Date', 'Days'])

    ### Import options

    # Function to parse dates of '20090101' format
    f = lambda x: dt.datetime.strptime(x, '%Y%m%d')
    raw_options = pd.read_csv('../data/options.csv',
                              converters={'Expiration': f})

    # Function to convert days to internal timedelta format
    f_delta = lambda x: dt.timedelta(days=int(x))
    raw_options['Date'] = raw_options['Expiration'] \
        - raw_options['Days'].map(f_delta)
    # Convert integer strikes to float!
    # Otherwise it may lead to accumulation of errors.
    raw_options['Strike'] = raw_options['Strike'].astype(float)

    ### Do some cleaning and indexing

    # Since VIX is computed for the date of option quotations,
    # we do not really need Expiration
    options = raw_options.set_index(['Date','Days','Strike']).drop('Expiration', axis = 1)

    # Do some renaming and separate calls from puts
    calls = options[['Call Bid','Call Ask']].rename(columns = {'Call Bid' : 'Bid', 'Call Ask' : 'Ask'})
    puts = options[['Put Bid','Put Ask']].rename(columns = {'Put Bid' : 'Bid', 'Put Ask' : 'Ask'})

    # Add a column indicating the type of the option
    calls['CP'], puts['CP'] = 'C', 'P'

    # Merge calls and puts
    options = pd.concat([calls, puts])

    # Reindex and sort
    options = options.reset_index().set_index(['Date','Days','CP','Strike']).sort_index()

    ### Compute bid/ask average

    # This step is used further to filter out in-the-money options.

    options['Premium'] = (options['Bid'] + options['Ask']) / 2
    options2 = options[options['Bid'] > 0]['Premium'].unstack('CP')

    ### Determine minimum difference

    # In[6]:

    # Find the absolute difference
    options2['CPdiff'] = (options2['C'] - options2['P']).abs()
    # Mark the minimum for each date/term
    options2['min'] = options2['CPdiff'].groupby(level=['Date','Days']).transform(lambda x: x == x.min())

    print(options2.ix[(dt.date(2009,1,1),9,910):(dt.date(2009,1,1),9,930)])

    ### Compute forward price

    # Leave only at-the-money optons
    df = options2[options2['min'] == 1].reset_index()
    # Merge with risk-free rate
    df = pd.merge(df, yields.reset_index(), how = 'left')

    # Compute the implied forward
    df['Forward'] = df['CPdiff'] * np.exp(df['Rate'] * df['Days'] / 36500)
    df['Forward'] += df['Strike']
    forward = df.set_index(['Date','Days'])[['Forward']]

    ### Compute at-the-money strike

    # Merge options with implied forward price
    left = options2.reset_index().set_index(['Date','Days'])
    df = pd.merge(left, forward, left_index = True, right_index = True)
    # Compute at-the-money strike
    mid_strike = df[df['Strike'] < df['Forward']]['Strike'].groupby(level = ['Date','Days']).max()
    mid_strike = pd.DataFrame({'Mid Strike' : mid_strike})

    ### Separate out-of-the-money calls and puts

    # Go back to original data and reindex it
    left = options.reset_index().set_index(['Date','Days']).drop('Premium', axis = 1)
    # Merge with at-the-money strike
    df = pd.merge(left, mid_strike, left_index = True, right_index = True)
    # Separate out-of-the-money calls and puts
    P = (df['Strike'] <= df['Mid Strike']) & (df['CP'] == 'P')
    C = (df['Strike'] >= df['Mid Strike']) & (df['CP'] == 'C')
    puts, calls = df[P], df[C]

    ### Remove all quotes after two consequtive zero bids

    # Indicator of zero bid
    calls['zero_bid'] = (calls['Bid'] == 0).astype(int)
    # Accumulate number of zero bids starting at-the-money
    calls['zero_bid_accum'] = calls.groupby(level = ['Date','Days'])['zero_bid'].cumsum()

    # Sort puts in reverse order inside date/term
    puts = puts.groupby(level = ['Date','Days']).apply(lambda x: x.sort_index(by = 'Strike', ascending = False))
    # Indicator of zero bid
    puts['zero_bid'] = (puts['Bid'] == 0).astype(int)
    # Accumulate number of zero bids starting at-the-money
    puts['zero_bid_accum'] = puts.groupby(level = ['Date','Days'])['zero_bid'].cumsum()

    # Merge puts and cals
    options3 = pd.concat([calls, puts]).reset_index()
    # Throw away bad stuff
    options3 = options3[(options3['zero_bid_accum'] < 2) & (options3['Bid'] > 0)]

    # Compute option premium as bid/ask average
    options3['Premium'] = (options3['Bid'] + options3['Ask']) / 2
    options3 = options3.set_index(['Date','Days','CP','Strike'])['Premium'].unstack('CP')

    ### Compute out-of-the-money option price

    # Merge wth at-the-money strike price
    left = options3.reset_index().set_index(['Date','Days'])
    df = pd.merge(left, mid_strike, left_index = True, right_index = True)

    # Conditions to separate out-of-the-money puts and calls
    condition1 = df['Strike'] < df['Mid Strike']
    condition2 = df['Strike'] > df['Mid Strike']
    # At-the-money we have two quotes, so take the average
    df['Premium'] = (df['P'] + df['C']) / 2
    # Remove in-the-money options
    df['Premium'].ix[condition1] = df['P'].ix[condition1]
    df['Premium'].ix[condition2] = df['C'].ix[condition2]

    options4 = df[['Strike','Mid Strike','Premium']].copy()

    ### Compute difference between adjoining strikes

    # Note that the strikes must be floats!!! In Python 3/2=1 and 3./2=1.5!

    def f(group):
        new = group.copy()
        new.iloc[1:-1] = np.array((group.iloc[2:] - group.iloc[:-2]) / 2)
        new.iloc[0] = group.iloc[1] - group.iloc[0]
        new.iloc[-1] = group.iloc[-1] - group.iloc[-2]
        return new

    options4['dK'] = options4.groupby(level = ['Date','Days'])['Strike'].apply(f)

    ### Compute contribution of each strike

    # Merge with risk-free rate
    contrib = pd.merge(options4, yields, left_index = True, right_index = True).reset_index()

    contrib['sigma2'] = contrib['dK'] / contrib['Strike'] ** 2
    contrib['sigma2'] *= contrib['Premium'] * np.exp(contrib['Rate'] * contrib['Days'] / 36500)

    ### Compute each preiod index

    # Sum up contributions from all strikes
    sigma2 = contrib.groupby(['Date','Days'])[['sigma2']].sum() * 2

    # Merge at-the-money strike and implied forward
    sigma2['Mid Strike'] = mid_strike
    sigma2['Forward'] = forward

    # Compute variance for each term
    sigma2['sigma2'] -= (sigma2['Forward'] / sigma2['Mid Strike'] - 1) ** 2
    sigma2['sigma2'] /= sigma2.index.get_level_values(1).astype(float) / 365
    sigma2 = sigma2[['sigma2']]

    print(sigma2.head())


    ### Compute interpolated index

    # This function determines near- and next-term if there are several maturities in the data
    def f(group):
        days = np.array(group['Days'])
        sigma2 = np.array(group['sigma2'])

        if days.min() <= 30:
            T1 = days[days <= 30].max()
        else:
            T1 = days.min()

        T2 = days[days > T1].min()

        sigma_T1 = sigma2[days == T1][0]
        sigma_T2 = sigma2[days == T2][0]

        return pd.DataFrame([{'T1' : T1, 'T2' : T2, 'sigma2_T1' : sigma_T1, 'sigma2_T2' : sigma_T2}])

    two_sigmas = sigma2.reset_index().groupby('Date').apply(f).groupby(level = 'Date').first()

    print(two_sigmas.head())


    ### Interpolate the VIX

    df = two_sigmas.copy()

    for t in ['T1','T2']:
        # Convert to fraction of the year
        df['days_' + t] = df[t].astype(float) / 365
        # Convert to miutes
        df[t] = (df[t] - 1) * 1440. + 510 + 930

    df['sigma2_T1'] = df['sigma2_T1'] * df['days_T1'] * (df['T2'] - 30. * 1440.)
    df['sigma2_T2'] = df['sigma2_T2'] * df['days_T2'] * (30. * 1440. - df['T1'])
    df['VIX'] = ((df['sigma2_T1'] + df['sigma2_T2']) / (df['T2'] - df['T1']) * 365. / 30.) ** .5 * 100

    return df[['VIX']]


if __name__ == '__main__':

    vix = whitepaper()
    print(vix)
