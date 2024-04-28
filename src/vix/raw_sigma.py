# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:55:06 2012

@author: khrapovs
"""
import pandas as pd
import numpy as np

' ------------------------------------------------------------------ '
' Function to compute raw sigma2 '
def raw_sigma(group):
    global p
    p1 = np.max((int(group['Maturity'].min()), p))
    ' T1 is near term maturity '
    if p <= 7:
        T1 = group['Maturity'][group['Maturity'] >= 7].min()
    else:
        T1 = group['Maturity'][group['Maturity'] <= p1].max()
    ' T2 is next term maturity '
    p2 = int(np.max((p, T1)))
    T2 = group['Maturity'][group['Maturity'] > p2].min()
    
    ' Initialize weights for near and next term variances '
    group['weight'] = np.nan
    group['weight'][group['Maturity'] == T1] = float((T2 - p) / (T2 - T1))
    group['weight'][group['Maturity'] == T2] = float((p - T1) / (T2 - T1))
    ' Work with near and next term separately '
    group1 = group[group['Maturity'] == T1]
    group2 = group[group['Maturity'] == T2]
    
    ' For near and next term separately '
    for g in (group1, group2):

        ' Find strikes where both puts and calls are traded '
        strikes = np.intersect1d(g['strike_price'][g['cp_flag'] == 'C'], \
            g['strike_price'][g['cp_flag'] == 'P'])

        ' Find calls and puts traded at the same strikes '
        Call_K = np.array(g['Quote'][(g['cp_flag'] == 'C') \
            & (g['strike_price'].isin(strikes))])
        Put_K = np.array(g['Quote'][(g['cp_flag'] == 'P') \
            & (g['strike_price'].isin(strikes))])
        
        ' Try to find minimum difference between prices '
        try:
            i = np.abs(Call_K - Put_K).argmin()
        except:
            g['sigma2'] = np.nan
            continue
        
        ' Forward index according to the formula '
        g['Forward'] = strikes[i] \
            + np.exp( g['rate'] / 36500 * g['Maturity']) \
            * (Call_K[i] - Put_K[i])
        ' First strike below forward '
        g['K0'] = g['strike_price'][g['strike_price'] \
            <= g['Forward']].max()
        
        ' Initialaze contribution vector '
        g['KP'] = np.nan
        ' Conditions for out-of-the-money options '
        left = (g['strike_price'] <= g['K0']) & (g['cp_flag'] == 'P')
        right = (g['strike_price'] >= g['K0']) & (g['cp_flag'] == 'C')
        ' Leave only out-of-the-money options '
        g['KP'][left] = g['Quote'][left]
        g['KP'][right] = g['Quote'][right]
        g['low'], g['high'], g['dK'] = np.nan, np.nan, np.nan
        
        ' Find distance between strikes '
        for cp in (left, right):
            g['low'][cp] = g['strike_price'][cp].shift(1)
            g['high'][cp] = g['strike_price'][cp].shift(-1)
            g['dK'][cp] = (g['high'][cp] - g['low'][cp]) / 2
            try:
                cond1 = cp & g['low'].isnull()
                cond2 = cp & g['high'].isnull()
                ' Distance between strikes on the edges '
                g['dK'][cond1] = g['high'][cond1] - g['strike_price'][cond1]
                g['dK'][cond2] = g['strike_price'][cond2] - g['low'][cond2]
            except:
                g['sigma2'] = np.nan
                continue

        ' Compute raw sigma2 for each maturity '
        g['sigma2'] = 2 * g['KP'] * g['dK'] \
            / g['strike_price'] ** 2 \
            * np.exp( g['rate'] / 36500 * g['Maturity'])
        g['FK0'] = (g['Forward'] / g['K0'] - 1) ** 2
    
    ' Concatenate near/next term optios '
    group = pd.concat([group1, group2])
    ' Apply near/next term weights '
    group['sigma2'] = group['sigma2'] * group['weight']
    ' Leave only positive variances '
    group = group[group['sigma2'].notnull()]
    
    return group