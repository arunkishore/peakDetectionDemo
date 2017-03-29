#http://tanzimsaqib.com/time-series-101
#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic
np.set_printoptions(threshold=np.inf)


FILE_NAME = '/Users/ramakrishnanak/Downloads/GlobalLandTemperatures-2/GlobalLandTemperaturesByCountry.csv'

def test_stationarity(df):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(df)
    indices = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    output = pd.Series(dftest[0:4], index=indices)
    for key, value in dftest[4].items():
        output['Critical Value (%s)' % key] = value
    print (output)


def main():

    df = pd.read_csv(FILE_NAME, sep=',', skipinitialspace=True, encoding='utf-8')
    df = df.drop('AverageTemperatureUncertainty', axis=1)
    df = df[df.Country == 'Canada']
    df = df.drop('Country', axis=1)
    df.index = pd.to_datetime(df.dt)
    df = df.drop('dt', axis=1)
    df = df.ix['1900-01-01':]
    df = df.sort_index()

    # Display AT
    df.AverageTemperature.fillna(method='pad', inplace=True)
    mp.plot(df.AverageTemperature)
    mp.show()

    # Rolling Mean
    df.AverageTemperature.plot.line(style='b', legend=True, grid=True, label='Avg. Temperature (AT)')
    ax = df.AverageTemperature.rolling(window=12).mean().plot.line(style='r', legend=True, label='Mean AT')
    ax.set_xlabel('Date')
    mp.legend(loc='best')
    mp.title('Weather timeseries visualization')
    mp.show()

    test_stationarity(df.AverageTemperature)

    res = arma_order_select_ic(df.AverageTemperature, ic=['aic', 'bic'], trend='nc',
              max_ar=10, max_ma=10, fit_kw={'method': 'css-mle'})
    print (res)

    # Fit the model
    ts = pd.Series(df.AverageTemperature, index=df.index)
    model = ARMA(ts, order=(5, 6))
    results = model.fit(trend='nc', method='css-mle')
    print(results.summary2())

    # Plot the model
    fig, ax = mp.subplots(figsize=(10, 8))
    fig = results.plot_predict('01/01/2003', '12/01/2023', ax=ax)
    ax.legend(loc='lower left')
    mp.title('Weather Time Series prediction')
    mp.show()

    predictions = results.predict('01/01/2003', '12/01/2023')
    # You can manipulate/print the predictions after this

if __name__ == "__main__":
    main()


