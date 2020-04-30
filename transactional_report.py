#importing necessary modules

import pandas as pd
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plotter
import statsmodels.api as s
import matplotlib
from pylab import rcParams
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import datetime as d

warnings.filterwarnings("ignore")#never print matching warnings

#We are using 'fivethirtyeight' in-built style of matplotlib.style module

plotter.style.use('fivethirtyeight')

#Setting default matplotlib values and reading data from input file

matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['text.color'] = 'k'
data=pd.read_excel("C:\\Users\\Vijayant\\Desktop\\treport11.xls")
print(data.head(20))

#There are many categories in the transactional report data, we start from time series analysis and forecasting for Entity
#ENTY Filetype

enty = data.loc[data['Filetype'] == 'ENTY']

#sorting the rows with respect to dates and assigning the date column as the index of the records
#indexing with time series data

enty['Date'].min(),enty['Date'].max()
enty=enty.sort_values('Date')
nullcheck=enty.isnull().sum()
print (nullcheck)

#indexing with time series

enty = enty.set_index('Date')
print(enty.index)

#The current datetime data may be complex and tricky to handle, so, we will use the averages daily ENTY Linecount value for
#that month instead, and we are using the start of each month as the timestamp.

y_plot = enty['Linecount'].resample('W').mean()
#To check mean values of Linecount of ENTY filetype from 2014.

print(y_plot[:])
y_plot.plot(figsize=(15, 6))
plotter.show()


#We can also visualize our data using a method called time-series decomposition that allows us to decompose our
# time series into three distinct components: trend, seasonality, and noise.

X = y_plot
X=y_plot.fillna(0)
total_size = int(len(X) * 0.66)

train_set, test_set = X[0:total_size], X[total_size:len(X)]
train_data = [x for x in train_set]
predicted_data = list()
for test in range(len(test_set)):
    if test_set[test]==None:
        continue
    model = ARIMA(train_data, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    a = output[0]
    predicted_data.append(a)
    observed = test_set[test]
    train_data.append(observed)

    print('Week=%s, expected=%f ,predicted=%f, ' % (y_plot.index[test],observed,a))
mse_error = mean_squared_error(test_set, predicted_data)
print('Test Mean Squared Error: %.3f' % mse_error)

y_plot = enty['Linecount'].resample('W').mean().fillna(0)
rcParams['figure.figsize'] = 18, 8

decomposition = s.tsa.seasonal_decompose(y_plot, model='additive')
fig = decomposition.plot()
plotter.show()

#Time series forecasting using ARIMA model

#Here I have tried to apply one of the most commonly used method for time-series forecasting, known as ARIMA, which stands
# for Autoregressive Integrated Moving Average.

#ARIMA models are denoted with notation ARIMA(p, d, q). The three parameters account for seasonality, trend, and noise in data respectively.

p = d = q = range(2)
prod = list(itertools.product(p, d, q))
seasonal_prod = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

'''print('Examples showing the parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(prod[1], seasonal_prod[1]))
print('SARIMAX: {} x {}'.format(prod[1], seasonal_prod[2]))
print('SARIMAX: {} x {}'.format(prod[2], seasonal_prod[3]))
print('SARIMAX: {} x {}'.format(prod[2], seasonal_prod[4]))'''

#Here we are selecting parameter for our ENTY Filetype ARIMA Time Series Model.
# Our goal here is to use a “grid search” to find the optimal set of parameters that yields the best performance for our model.

for p in prod:
    for p_seasonal in seasonal_prod:
        try:
            mod = s.tsa.statespace.SARIMAX(y_plot,order=p,seasonal_order=p_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            out = mod.fit()
            #print('ARIMA{}x{}12 - AIC:{}'.format(p, p_seasonal,out.aic))

        except:
            continue

#SARIMAX(1, 1, 1)x(1, 1, 0, 12) yields the lowest AIC (The Akaike Information Critera )value of 739.329. Therefore we are choosing
# this to be optimal option.

#Now we will fit the ARIMA model

mod=s.tsa.statespace.SARIMAX(y_plot,order=(1, 1, 1),seasonal_order=(1, 1, 0, 12),enforce_stationarity=False,enforce_invertibility=False)
out = mod.fit()
print(out.summary().tables[1])

#We are always running model diagnostics to investigate any unusual behavior.

out.plot_diagnostics(figsize=(16, 8))
plotter.show()

#Forecast validation

predict = out.get_prediction(start=pd.to_datetime('2013-06-02'), dynamic=False)
predict_ci = predict.conf_int()

x = y_plot['2013':].plot(label='observed values from the input file')
predict.predicted_mean.plot(ax=x, label='Forecast validation', alpha=.7, figsize=(14, 7))

x.fill_between(predict_ci.index,predict_ci.iloc[:,0],predict_ci.iloc[:,1], color='k', alpha=.2)
x.set_xlabel('Date')
x.set_ylabel('LKLT')
plotter.legend()
plotter.show()

#Now visualizing forecast for the required number of steps.

y_forecasted = predict.predicted_mean
y_truth = y_plot['2013-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
predict_uc = out.get_forecast(steps=50)
predict_ci = predict_uc.conf_int()

x1 = y_plot.plot(label='actual value observed', figsize=(14, 7))
predict_uc.predicted_mean.plot(ax=x1, label='Forecasted value')
x1.fill_between(predict_ci.index,
                predict_ci.iloc[:, 0],
                predict_ci.iloc[:, 1], color='k', alpha=.25)
x1.set_xlabel('Date')
x1.set_ylabel('ENTY')
plotter.legend()
plotter.show()
#Plotting the input records in the pie chart according to USA and non-USA LINK and ENTITY files.

country=data['Country']
date=data['Date']
file=data['Loadfile']
arrtime=data['Rectime']
sttime=data['Starttime']
comptime=data['Endtime']
ftype=data['Filetype']
lcount=data['Linecount']
dur=data['Duration']
usaentitycount=0
usalinkcount=0
nonusalinkcount=0
nonusaentitycount=0
totalusarecords=0
totalnonusarecords=0
for i in range(len(data)):
    if (country[i]=='USA' and ftype[i]=='ENTY'):
        usaentitycount=usaentitycount+1
    elif (country[i]=='USA' and ftype[i]=='LKLT'):
        usalinkcount=usalinkcount+1
    elif (country[i]!='USA' and ftype[i]=='ENTY'):
        nonusaentitycount=nonusaentitycount+1
    else:
        nonusalinkcount=nonusalinkcount+1

for j in range(len(data)):
    if (country[j] == 'USA'):
        totalusarecords=totalusarecords+lcount[j]
    else:
        totalnonusarecords = totalnonusarecords + lcount[j]
print("Total count of USA entity file is: ",usaentitycount)
print("Total count of USA link file is: ",usalinkcount)
print("Total count of non-USA entity file is: ",nonusaentitycount)
print("Total count of non-USA link file is: ",nonusalinkcount)
print("Total records received for USA is: ",totalusarecords)
print("Total records received for non-USA is: ",totalnonusarecords)
names='USA Entity Loadfiles','USA Link Loadfiles','Non-USA Entity Loadfile','Non-USA Link Loadfile'
values=[usaentitycount,usalinkcount,nonusaentitycount,nonusalinkcount]
figureObject, axesObject = plotter.subplots()
axesObject.pie(values,labels=names,autopct='%1.2f',startangle=90)
plotter.title("Analysis of USA and non-USA Transition loadfiles", bbox={'facecolor':'0.8', 'pad':5})
axesObject.axis('equal')
plotter.show()

