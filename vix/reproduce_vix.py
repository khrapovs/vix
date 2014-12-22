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

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"


if __name__ == '__main__':
    pass
