# VIX and related volatility indices

[Jupyter notebook](notebooks/Replicate_VIXwite.ipynb)

This notebook shows how to reproduce the VIX given the data in CBOE White Paper (http://www.cboe.com/micro/vix/vixwhite.pdf). The code works for any option data set, not only one day as in the White Paper. The option data for this example is exactly the same as in the Appendix 1 of the White Paper.

Given are the prices $C_{i}$, $i\in \lbrace 0,\ldots,n \rbrace $, of a series of European call options on the index with fixed maturity date $T$ and exercise prices $K_{i}$, $i\in\lbrace 0,\ldots,n\rbrace$, as well as the prices $P_{i}$, $i\in\lbrace 0,\ldots,n\rbrace$, of a series of European put options on the index with the same maturity date $T$ and exercise prices $K_{i}$. Let further hold $K_{i}<K_{i+1}$ for all $i\in\lbrace 0,\ldots,n-1\rbrace$.

The VIX itself is

$$VIX=100\cdot\sqrt{V^{2}},$$

where $V$ is explained below.

Since there are days when there no options with precisely 30 days to expiration, we have to interpolate between near-term index and next-term index:

$$V^{2}=\left[T_{1}\sigma_{1}^{2}\left(\frac{N_{T_{2}}-N_{30}}{N_{T_{2}}-N_{T_{1}}}\right)+T_{2}\sigma_{2}^{2}\left(\frac{N_{30}-N_{T_{1}}}{N_{T_{2}}-N_{T_{1}}}\right)\right]\frac{365}{30}$$

with each $\sigma_{i}^{2}$ computed according to

$$\sigma^{2}=\frac{2}{T}\sum_{i=0}^{n}\frac{\Delta K_{i}}{K_{i}^{2}}e^{rT}M_{i}-\frac{1}{T}\left(\frac{F}{K_{\star}}-1\right)^{2},$$

where the distance between strikes is

$$\Delta K_{i}	=	\begin{cases}
K_{1}-K_{0}, & i=0\\
\frac{1}{2}\left(K_{i+1}-K_{i-1}\right), & i=1,\ldots,n-1\\
K_{n}-K_{n-1}, & i=n
\end{cases}$$

the out-of-the-money option premium is

$$M_{i}	=	\begin{cases}
P_{i}, & K_{i} < K_{\star}\\
\frac{1}{2}\left(P_{i}+C_{i}\right), & K_{i}=K_{\star}\\
C_{i}, & K_{i} > K_{\star}
\end{cases}$$

at-the-money strike price is

$$K_{\star}	=	\max\lbrace K_{i} < F\rbrace ,$$

forward price extracted from put-call parity:

$$F	=	K_{j}+e^{rT}\left|C_{j}-P_{j}\right|,$$

with

$$j=\min\lbrace \left|C_{i}-P_{i}\right|\rbrace ,$$

and finally $r$ is the constant risk-free short rate appropriate for maturity $T$.

## Installation

```shell
pip install .
```

## Contribute

Create a virtual environment and activate it:
```shell
python -m venv venv
source venv/bin/activate
```
Install development dependencies:
```shell
pip install -e .
```
