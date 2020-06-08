# Cauldrons Quantitative Research Library (cqrlib)

Research tools for educational quantitative research "Project Cauldrons".

Created for easier research, this repositary is built mainly using multiprocessing for parallel computering & pandas.
As much as possible, we would prefer to use these as ou core modules to ensure code consistancy, easier maintenance (version migration) as well as taking it as an opportunity to improve Python knowledge.

cqrlib was created purely for educational purpose, these codes however are meant for quantitative strategy development/ implementation.

Feel free to contribute by completing any issues filed.

Please ensure you have the below modules installed/ upgrade in conda enviroment or pip:

* Python 3.7.4
* numpy 1.17.3
* numba 0.49.1
* pandas 1.0.3
* matplotlib 3.1.1
* sklearn 0.23.1

If you are interested in understanding how the codes work, I provided a notebook tutorial answers for easier understanding.

[AFML juypter notebook](https://github.com/boyboi86/AFML)

**Disclaimer**
Due to lack of resource, cqrlib can only provide most but not all algorithms found within Advances in Financial Machine Learning (AFML Chapter 2).

## Mean-Reversion/ Futures Related

Most of the logic can be found in Advance in Financial Machine Learning. 

![logo](https://media.wiley.com/product_data/coverImage300/89/11194820/1119482089.jpg)

The textbook itself is meant for graduate studies writtern by Dr Macros Lopez De Prado.

I highly recommend you to purchase this book to bridge the knowledge gap.

The mathematics behind what was taught in the book are actually sounded and practical (Considered white box).
Every procedure that was taught is specifically addressing concerns using Machine-learning within Financial industry.

1. Either to improve statistic properties
2. Provide better machine-learning results
3. Code optimization

Most of the code snippets written in the book was done using Python 2 & Pandas before stable release.

The codes found in this repository is different from the book so that it will be compatible in Python 3 & Pandas 1.0.3 (Rather than Numpy).
Hopefully new "quants" can relate to the textbook & appreciate the theory behind these codes.

> "Every successful investment strategy is always supported by an equally sound theory that is practical,
> you either prove with results or you prove the theory first.
>
> Either way one will always lead to another."
>
> &mdash; The cashier who initiated "Project Cauldrons"

Disclaimer: The code base is still messy since it is still under development stage and due to lack of high frequency data samples, not all the techniques taught will be implemented.

* Standard Data Bars
* Imbalance Bars
* Meta-Labels
* Fractional Differentiate
* Sequential bootstrapping

For option related codes, I haven't develop as much lately. (Not extremely useful) 

I am looking forward to build option related strategies based on "Volatility Trading" written by a fellow Mensian, Dr Euan Sinclair.

![logo](https://media.wiley.com/product_data/coverImage300/37/11183471/1118347137.jpg)

## Volatility/ Option Related

* Black-Scholes-Merton Option Theoretical pricing, IV, Delta, Gamma, Vega
* Premium Calculator
* Probaility Calculator
* Probability of Profit(POP)
* Volatility Percentile Score
* Parkinson Volatility
* Yang-Zhang Volatility
* EWMA Daily Volatility Estimator
* Garman Class Volatility
