# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 18:02:32 2012

@author: skhrapov

Extract SPX prices, returns, option prices, implied volatilities, etc

"""

import datetime as dt
import zipfile
from io import StringIO

import numpy as np
import pandas as ps

# """ ------------------------------------------------------------------ """
# """ Import real VIX """
f = lambda x: dt.datetime.strptime(x, '%Y-%m-%d').date() if x != '' else x
realdata = ps.read_csv('Data_cleaning/SPX_RV_VIX.csv', converters={'Date': f})
realVIX = ps.Series(realdata['VIX'], index=ps.Index(realdata['Date']))

# """ ------------------------------------------------------------------ """
# """ Import yield curve """

zf = zipfile.ZipFile('WRDS/yield_curve.zip', 'r')
name = zf.namelist()[0]
data = StringIO.StringIO(zf.read(name))
f = lambda x: dt.datetime.strptime(x, '%Y%m%d').date() if x != '' else x
yields = ps.read_csv(data, converters={'date': f})

# """ ------------------------------------------------------------------ """
# """ Import options """

df = ps.read_csv('options.csv', converters={'date': f, 'exdate': f})
df['Maturity'] = (df['exdate'] - df['date']).map(lambda x: x.days)

df['Quote'] = (df['best_bid'] + df['best_offer']) / 2
df['strike_price'] = df['strike_price'] / 1e3

# """ ------------------------------------------------------------------ """
# """ Merge table with risk-free rates """
table = ps.DataFrame()
for d in np.unique(df['date']):
    dfy = yields[yields['date'] == d]
    dfc = df[df['date'] == d]

    f = lambda x: dfy['days'][dfy.index[(dfy['days'] - x).abs().argmin()]]

    dfc['ydays'] = dfc['Maturity'].map(f)
    dfc = ps.merge(dfc, dfy, left_on=['date', 'ydays'], right_on=['date', 'days'])
    table = table.append(dfc, ignore_index=True)


# """ ------------------------------------------------------------------ """
# """ Function to compute raw sigma2 """
def f(group):
    p1 = np.max((int(group['Maturity'].min()), p))
    # """ T1 is near term maturity """
    if p <= 7:
        T1 = group['Maturity'][group['Maturity'] >= 7].min()
    else:
        T1 = group['Maturity'][group['Maturity'] <= p1].max()
    # """ T2 is next term maturity """
    p2 = int(np.max((p, T1)))
    T2 = group['Maturity'][group['Maturity'] > p2].min()

    # """ Initialize weights for near and next term variances """
    group['weight'] = np.nan
    group['weight'][group['Maturity'] == T1] = float((T2 - p) / (T2 - T1))
    group['weight'][group['Maturity'] == T2] = float((p - T1) / (T2 - T1))
    # """ Work with near and next term separately """
    group1 = group[group['Maturity'] == T1]
    group2 = group[group['Maturity'] == T2]

    # """ For near and next term separately """
    for g in (group1, group2):

        # """ Find strikes where both puts and calls are traded """
        strikes = np.intersect1d(g['strike_price'][g['cp_flag'] == 'C'], g['strike_price'][g['cp_flag'] == 'P'])

        # """ Find calls and puts traded at the same strikes """
        Call_K = np.array(g['Quote'][(g['cp_flag'] == 'C') & (g['strike_price'].isin(strikes))])
        Put_K = np.array(g['Quote'][(g['cp_flag'] == 'P') & (g['strike_price'].isin(strikes))])

        # """ Try to find minimum difference between prices """
        try:
            i = np.abs(Call_K - Put_K).argmin()
        except:
            g['sigma2'] = np.nan
            continue

        # """ Forward index according to the formula """
        g['Forward'] = strikes[i] + np.exp(g['rate'] / 36500 * g['Maturity']) * (Call_K[i] - Put_K[i])
        # """ First strike below forward """
        g['K0'] = g['strike_price'][g['strike_price'] <= g['Forward']].max()

        # """ Initialaze contribution vector """
        g['KP'] = np.nan
        # """ Conditions for out-of-the-money options """
        left = (g['strike_price'] <= g['K0']) & (g['cp_flag'] == 'P')
        right = (g['strike_price'] >= g['K0']) & (g['cp_flag'] == 'C')
        # """ Leave only out-of-the-money options """
        g['KP'][left] = g['Quote'][left]
        g['KP'][right] = g['Quote'][right]
        g['low'], g['high'], g['dK'] = np.nan, np.nan, np.nan

        # """ Find distance between strikes """
        for cp in (left, right):
            g['low'][cp] = g['strike_price'][cp].shift(1)
            g['high'][cp] = g['strike_price'][cp].shift(-1)
            g['dK'][cp] = (g['high'][cp] - g['low'][cp]) / 2
            try:
                # """ Distance between strikes on the edges """
                g['dK'][cp & g['low'].isnull()] = g['high'][cp & g['low'].isnull()] - g['strike_price'][
                    cp & g['low'].isnull()]
                g['dK'][cp & g['high'].isnull()] = g['strike_price'][cp & g['high'].isnull()] - g['low'][
                    cp & g['high'].isnull()]
            except:
                g['sigma2'] = np.nan
                continue

        # """ Compute raw sigma2 for each maturity """
        g['sigma2'] = 2 * g['KP'] * g['dK'] / g['strike_price'] ** 2 * np.exp(g['rate'] / 36500 * g['Maturity'])
        g['FK0'] = (g['Forward'] / g['K0'] - 1) ** 2

    # """ Concatenate near/next term optios """
    group = ps.concat([group1, group2])
    # """ Apply near/next term weights """
    group['sigma2'] = group['sigma2'] * group['weight']
    # """ Leave only positive variances """
    group = group[group['sigma2'].notnull()]

    return group


table = table.sort_index(by=['date', 'strike_price', 'cp_flag', 'Maturity'])
gtable = table.groupby('date')

period = [30]
myVIX = ps.DataFrame()
for p in period:
    table = gtable.apply(f)[['date', 'sigma2', 'FK0']]
    gtable = table.groupby('date')
    sigmas = gtable['sigma2'].sum() - gtable['FK0'].mean()
    intVIX = (sigmas * 365 / p) ** .5 * 100
    if myVIX.empty:
        myVIX = ps.DataFrame(intVIX, columns=[str(p)])
    else:
        myVIX[str(p)] = intVIX

# """ add real data to the table """
myVIX['VIX'] = realVIX

# """ ------------------------------------------------------------------ """
# """ Plot data """
# myVIX.plot()
# plt.gcf().autofmt_xdate()
# plt.show()
