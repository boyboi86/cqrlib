<<<<<<< HEAD
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


=======
<div align="center">
  <img src="https://raw.githubusercontent.com/hudson-and-thames/mlfinlab/master/.github/logo/hudson_and_thames_logo.png" height="300"><br>
</div>

-----------------

# Research Repo

Contains all the Jupyter Notebooks used in our research.

All of the research we do in these notebooks is on the full tick history dataset from Tick Data LLC, but not provided because of royalty fees.  The data can be purchased for about $750 US Dollars for the full history of a single ticker.

We do provide a 2 year sample on tick, volume, and dollar bars to help the community get started. 

## Contributing

<div align="center">
  <img src="https://raw.githubusercontent.com/hudson-and-thames/research/master/Chapter3/readme_image.png" height="350"><br>
</div>

Our hope is that the sample data and notebooks will enable the community to build on the research and contribute to the open source community. 

A good place to start for new users is to use the data provided to answer the questions at the back of the chapters in Advances in Financial Machine Learning.

Please review the [Guidelines](https://github.com/hudson-and-thames/research/tree/master/Guidelines.md) for research

### Sample Data

The following [folder](https://github.com/hudson-and-thames/research/tree/master/Sample-Data) contains 2 years sample data on S&P500 Emini Futures, for the period 2015-01-01 to 2017-01-01.

Specifically the following data structures:
* Dollar Bars: Sampled every $70'000
* Volume Bars: Sampled every 28'000 contracts
* Tick Bars: Sampled every 2'800 ticks

## Installation

Recommended versions:
* Anaconda 3
* Python 3.6

### Installation for Mac OS X and Ubuntu Linux

1. Make sure you install the latest version of the Anaconda 3 distribution. To do this you can follow the install and update instructions found on this link: https://www.anaconda.com/download/#mac
2. Launch a terminal
3. Create a New Conda Environment. From terminal: ```conda create -n <env name> python=3.6 anaconda``` accept all the requests to install.
4. Now activate the environment with ```source activate <env name>```.
5. From Terminal: go to the directory where you have saved the file, example: cd Desktop/research/.
6. Install Python requirements, by running the command: ```pip install -r requirements.txt```
7. (Optional) Continue to Chapter-specific Installation 

### Installation for Windows

1. Download and install the latest version of [Anaconda 3](https://www.anaconda.com/distribution/#download-section)
2. Launch Anaconda Navigator
3. Click Environments, choose an environment name, select Python 3.6, and click Create
4. Click Home, browse to your new environment, and click Install under Jupyter Notebook
5. Launch Anaconda Prompt and activate the environment: ```conda activate <env name>```
6. From Anaconda Prompt: go to the directory where you have saved the file, example: cd Desktop/research/.
7. Install Python requirements, by running the command: ```pip install -r requirements.txt```
8. (Optional) Continue to Chapter-specific Installation 

### Chapter-specific Installation

We will create a symlink inside each of the Chapters for ease of dataset changes. You may change the symlink of `official_data` to your own dataset rather than using the 2 year sample; the format follows Tick Data LLC.

Create a symbolic link inside the Chapter folder to where you saved the official data:

``` cd Chapter3; ln -s ../Sample-Data official_data ```

## Additional Research Repo
BlackArbsCEO has a great repo based on de Prado's research. It covers many of the questions at the back of every chapter and was the first source on Github to do so. It has also been a good source of inspiration for our research.

* [Adv Fin ML Exercises](https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises)
>>>>>>> 5be8c6dacb3f3c9cb78f0d7f8668be5f4550f1c1
