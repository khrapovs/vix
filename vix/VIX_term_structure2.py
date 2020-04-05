# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 18:02:32 2012

@author: skhrapov

Extract SPX prices, returns, option prices, implied volatilities, etc

"""

import datetime as dt
import time
import zipfile
from io import StringIO

import numpy as np
import pandas as ps

# ' ------------------------------------------------------------------ '
# ' Import real VIX '
t1 = time.time()
f = lambda x: dt.datetime.strptime(x, '%Y-%m-%d').date() if x != '' else x
realdata = ps.read_csv('Data_cleaning/SPX_RV_VIX.csv',
                       converters={'Date': f})
realVIX = ps.Series(realdata['VIX'], index=ps.Index(realdata['Date']))
t2 = time.time()
print(' Import real VIX ', (t2 - t1) / 60)

# ' ------------------------------------------------------------------ '
# ' Import yield curve '
t1 = time.time()
zf = zipfile.ZipFile('WRDS/yield_curve.zip', 'r')
name = zf.namelist()[0]
data = StringIO(zf.read(name))
f = lambda x: dt.datetime.strptime(x, '%Y%m%d').date() if x != '' else x
yields = ps.read_csv(data, converters={'date': f})
t2 = time.time()
print(' Import yield curve ', (t2 - t1) / 60)

# ' ------------------------------------------------------------------ '
# ' Import options '
t1 = time.time()
zf = zipfile.ZipFile('WRDS/SP500_options_1996_2011.zip', 'r')
name = zf.namelist()[0]
data = StringIO(zf.read(name))

reader = ps.read_csv(data, chunksize=int(1e4),
                     converters={'date': f, 'exdate': f})
t2 = time.time()
print(' Import options ', (t2 - t1) / 60)

# ' ------------------------------------------------------------------ '
# ' Light filter chunk by chunk '
t1 = time.time()
df = ps.DataFrame()
i = 0
for chunk in reader:
    chunk = chunk.dropna(axis=0)
    chunk = chunk[(chunk['best_bid'] > 0) & (chunk['best_offer'] > 0)]
    # chunk = chunk[chunk['opp_volume'] > 0]
    df = df.append(chunk, ignore_index=True)
    i += 1
    if i >= 1e1:
        break

df['Maturity'] = (df['exdate'] - df['date']).map(lambda x: x.days)
df['Quote'] = (df['best_bid'] + df['best_offer']) / 2
df['strike_price'] = df['strike_price'] / 1e3

# ddf = df.sort_index(by = ['date', 'cp_flag', 'Maturity', 'strike_price'])
# gdf = ddf.groupby(['date', 'cp_flag', 'Maturity'])

t2 = time.time()
print(' Light filter chunk by chunk ', (t2 - t1) / 60)

# ' ------------------------------------------------------------------ '
# ' Merge table with risk-free rates '
t1 = time.time()
table = ps.DataFrame()
# ' For each date in options table '
for d in np.unique(df['date']):
    dd = d
    dfy = yields[yields['date'] == dd]
    # ' Make sure that both tables are not empty on the same dates '
    while dfy['date'].isnull().all():
        # ' Shift date one day back if no yields on the date '
        dd -= dt.timedelta(1)
        dfy = yields[yields['date'] == dd]

    dfc = df[df['date'] == d]

    f = lambda x: dfy['days'][dfy.index[(dfy['days'] - x).abs().argmin()]]

    dfc['ydays'] = dfc['Maturity'].map(f)
    dfc = ps.merge(dfc, dfy, left_on=['date', 'ydays'], right_on=['date', 'days'])
    table = table.append(dfc, ignore_index=True)

t2 = time.time()
print(' Merge table with risk-free rates ', (t2 - t1) / 60)


# ' ------------------------------------------------------------------ '
# ' Function to compute raw sigma2 '


def f(group):
    p1 = np.max((int(group['Maturity'].min()), p))
    # ' T1 is near term maturity '
    if p <= 7:
        T1 = group['Maturity'][group['Maturity'] >= 7].min()
    else:
        T1 = group['Maturity'][group['Maturity'] <= p1].max()
    # ' T2 is next term maturity '
    p2 = int(np.max((p, T1)))
    T2 = group['Maturity'][group['Maturity'] > p2].min()

    # ' Initialize weights for near and next term variances '
    group['weight'] = np.nan
    group['weight'][group['Maturity'] == T1] = float((T2 - p) / (T2 - T1))
    group['weight'][group['Maturity'] == T2] = float((p - T1) / (T2 - T1))
    # ' Work with near and next term separately '
    group1 = group[group['Maturity'] == T1]
    group2 = group[group['Maturity'] == T2]

    # ' For near and next term separately '
    for g in (group1, group2):

        # ' Find strikes where both puts and calls are traded '
        strikes = np.intersect1d(g['strike_price'][g['cp_flag'] == 'C'], g['strike_price'][g['cp_flag'] == 'P'])

        # ' Find calls and puts traded at the same strikes '
        Call_K = np.array(g['Quote'][(g['cp_flag'] == 'C') & (g['strike_price'].isin(strikes))])
        Put_K = np.array(g['Quote'][(g['cp_flag'] == 'P') & (g['strike_price'].isin(strikes))])

        # ' Try to find minimum difference between prices '
        try:
            i = np.abs(Call_K - Put_K).argmin()
        except:
            g['sigma2'] = np.nan
            continue

        # ' Forward index according to the formula '
        g['Forward'] = strikes[i] + np.exp(g['rate'] / 36500 * g['Maturity']) * (Call_K[i] - Put_K[i])
        # ' First strike below forward '
        g['K0'] = g['strike_price'][g['strike_price'] <= g['Forward']].max()

        # ' Initialaze contribution vector '
        g['KP'] = np.nan
        # ' Conditions for out-of-the-money options '
        left = (g['strike_price'] <= g['K0']) & (g['cp_flag'] == 'P')
        right = (g['strike_price'] >= g['K0']) & (g['cp_flag'] == 'C')
        # ' Leave only out-of-the-money options '
        g['KP'][left] = g['Quote'][left]
        g['KP'][right] = g['Quote'][right]
        g['low'], g['high'], g['dK'] = np.nan, np.nan, np.nan

        # ' Find distance between strikes '
        for cp in (left, right):
            g['low'][cp] = g['strike_price'][cp].shift(1)
            g['high'][cp] = g['strike_price'][cp].shift(-1)
            g['dK'][cp] = (g['high'][cp] - g['low'][cp]) / 2
            try:
                cond1 = cp & g['low'].isnull()
                cond2 = cp & g['high'].isnull()
                # Distance between strikes on the edges
                g['dK'][cond1] = g['high'][cond1] - g['strike_price'][cond1]
                g['dK'][cond2] = g['strike_price'][cond2] - g['low'][cond2]
            except:
                g['sigma2'] = np.nan
                continue

        # ' Compute raw sigma2 for each maturity '
        g['sigma2'] = 2 * g['KP'] * g['dK'] / g['strike_price'] ** 2 * np.exp(g['rate'] / 36500 * g['Maturity'])
        g['FK0'] = (g['Forward'] / g['K0'] - 1) ** 2

    # ' Concatenate near/next term options '
    group = ps.concat([group1, group2])
    # ' Apply near/next term weights '
    group['sigma2'] = group['sigma2'] * group['weight']
    # ' Leave only positive variances '
    group = group[group['sigma2'].notnull()]

    return group


table = table.sort_index(by=['date', 'strike_price', 'cp_flag', 'Maturity'])
dtable = table.groupby('date')

# period = [5,10,15,30,60,90,180]
period = [5]
myVIX, counts = ps.DataFrame(), ps.DataFrame()
for p in period:
    t1 = time.time()
    table = dtable.apply(f)[['date', 'sigma2', 'FK0']]
    gtable = table.groupby('date')
    sigmas = gtable['sigma2'].sum() - gtable['FK0'].mean()
    intVIX = (sigmas * 365 / p) ** .5 * 100
    if myVIX.empty:
        myVIX = ps.DataFrame(intVIX, columns=[str(p)])
        counts = ps.DataFrame(gtable['sigma2'].count(), columns=[str(p)])
    else:
        myVIX[str(p)] = intVIX
        counts[str(p)] = gtable['sigma2'].count()
    t2 = time.time()
    print(' Compute myVIX ', (t2 - t1) / 60)

# ' add real data to the table '
myVIX['VIX'] = realVIX
myVIX.to_csv('myVIX.csv', index_label='Date')

# ' ------------------------------------------------------------------ '
# ' Plot data '
# myVIX[['5', 'VIX']].plot()
# plt.gcf().autofmt_xdate()
# plt.show()
#
# counts['5'].plot()
# plt.gcf().autofmt_xdate()
# plt.show()
