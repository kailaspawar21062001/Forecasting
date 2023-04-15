# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 21:01:17 2023

@author: kailas
"""

1]PROBLEM


BUSINESS OBJECTIVE:-Forecast CocaCola Prices.



#Import Liabrary
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt


# Dataset
df=pd.read_excel("D:/data science assignment/Assignments/18.Forecasting/CocaCola_Sales_Rawdata.xlsx")

#EDA
df.info()
df.shape
df.head()
df.tail()
df.describe()

df['t']=np.arange(42)
df['t_square']=df['t'] * df['t']
df['log']=np.log(df.Sales)

p=df['Quarter'][0]
p[0:2]

df['fordum'] = 0

for i in range(42):
    p=df['Quarter'][i]
    df['fordum'][i]=p[0:2]
    
Quarter_dummies=pd.DataFrame(pd.get_dummies(df['fordum']))    
df1=pd.concat([df,Quarter_dummies],axis=1)    

#Visulaztion-Time Plot
df1.Sales.plot()

#Split the Dataset
train=df1.head(38)
test=df1.tail(4)


# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


*********DATA DRIVEN BASED*****
(MOVING AVERAGE BASED....)

# Moving Average for the time series
mv_pred = df["Sales"].rolling(4).mean()
mv_pred.tail(4)
MAPE(mv_pred.tail(4), test.Sales)


# Plot with Moving Averages
df.Sales.plot(label = "org")
for i in range(2, 9, 2):
    df["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)


# Time series decomposition is the process of separating data into its core components.
# Time series decomposition plot using Moving Average
decompose_ts_add = seasonal_decompose(df.Sales, model = "additive", period = 4)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

decompose_ts_mul = seasonal_decompose(df.Sales, model = "multiplicative", period = 4)
decompose_ts_mul.plot()

# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(df.Sales, lags = 4)
tsa_plots.plot_pacf(df.Sales, lags=4)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.


*****DATA DRIVEN BASED****
(EXPONENTIAL SMOOTHING BASED..)

#Simple Exponential Smoothing
ses_model = SimpleExpSmoothing(train["Sales"]).fit()
pred_ses = ses_model.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_ses, test.Sales) 

# Holt method 
hw_model = Holt(train["Sales"]).fit()
pred_hw = hw_model.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hw, test.Sales) 

# Holts winter exponential smoothing with additive seasonality and additive trend

hwe_model_add_add = ExponentialSmoothing(train["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hwe_add_add, test.Sales) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(train["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hwe_mul_add, test.Sales) 




************MODEL BASED******

#LINEAR
import statsmodels.formula.api as smf

linear_model=smf.ols('Sales ~ t',data=train).fit()
lin_predict=pd.Series(linear_model.predict(pd.DataFrame(test)))
rmse_lin=np.mean(np.sqrt((np.array(test['Sales']) - np.array(lin_predict)) **2))
rmse_lin


#Exponential

exp_model=smf.ols('log ~ t',data=train).fit()
exp_predict=pd.Series(exp_model.predict(pd.DataFrame(test)))
rmse_exp=np.mean(np.sqrt((np.array(test['Sales']) - np.array(np.exp(exp_predict))) **2))
rmse_exp

#Quadrartic

quad_model=smf.ols('Sales ~ t+t_square',data=train).fit()
quad_predict=pd.Series(quad_model.predict(pd.DataFrame(test)))
rmse_quad=np.mean(np.sqrt((np.array(test['Sales']) - np.array(quad_predict)) **2))
rmse_quad


#Additive Seasonality

add=smf.ols('Sales ~ Q1+Q2+Q3',data=train).fit()
add_predict=pd.Series(add.predict(pd.DataFrame(test[['Q1','Q2','Q3']])))
rmse_add=np.mean(np.sqrt((np.array(test['Sales']) - np.array(add_predict)) **2))
rmse_add


#Multiplicative Seasonality

mul=smf.ols('log ~ Q1+Q2+Q3',data=train).fit()
mul_predict=pd.Series(mul.predict(pd.DataFrame(test)))
rmse_mul=np.mean(np.sqrt((np.array(test['Sales']) - np.array(np.exp(mul_predict))) **2))
rmse_mul


#Additive Seasonality with Linear Trend.

add_linear=smf.ols('Sales ~ t+Q1+Q2+Q3',data=train).fit()
add_linear_predict=pd.Series(add_linear.predict(pd.DataFrame(test)))
rmse_add_linear=np.mean(np.sqrt((np.array(test['Sales']) - np.array(add_linear_predict)) **2))
rmse_add_linear


#Additive Seasonality with Quadratic Trend.

add_quad=smf.ols('Sales ~ t+t_square+Q1+Q2+Q3',data=train).fit()
add_quad_predict=pd.Series(add_quad.predict(pd.DataFrame(test)))
rmse_add_quad=np.mean(np.sqrt((np.array(test['Sales']) - np.array(add_quad_predict)) **2))
rmse_add_quad


#Multiplicative Seasonality Linear Trend.

mul_lin=smf.ols('log ~ t+Q1+Q2+Q3',data=train).fit()
mul_lin_predict=pd.Series(mul_lin.predict(pd.DataFrame(test)))
rmse_mul_lin=np.mean(np.sqrt((np.array(test['Sales']) - np.array(np.exp(mul_lin_predict))) **2))
rmse_mul_lin



#Multiplicative Seasonality Quadratic Trend.
mul_quad=smf.ols('log ~ t+t_square+Q1+Q2+Q3',data=train).fit()
mul_quad_predict=pd.Series(mul_quad.predict(pd.DataFrame(test)))
rmse_mul_quad=np.mean(np.sqrt((np.array(test['Sales']) - np.array(np.exp(mul_quad_predict))) **2))
rmse_mul_quad



data = {"MODEL":pd.Series(["rmse_add","rmse_add_linear","rmse_add_Quad","rmse_exp","rmse_lin","rmse_mul","rmse_Mul_lin","rmse_mul_quad"]),"RMSE_Values":pd.Series([rmse_add,rmse_add_linear,rmse_add_quad,rmse_exp,rmse_lin,rmse_mul,rmse_mul_lin,rmse_mul_quad])}
table_rmse = pd.DataFrame(data)
table_rmse

#Considering the above all value..I select (MAPE=1.502191616043862),which is 'LESS' among all models,
#so that's why,we use:- "Holts winter exponential smoothing with additive seasonality and additive trend" for forcasting.



################################################################################################################################################################################################################################################################################
2]Problem
    
    

BUSINESS OBJECTIVE::-
Forecast Airlines Passengers data set.




##importing the required libraries for Analysis

import pandas as pd
import numpy as np
from numpy import sqrt

from pandas import Grouper
from pandas import DataFrame

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.tsa.seasonal import seasonal_decompose
## with the help of this we will be able to creat graphs for the dfferent components of time series data 
#like trends, level, sesional components and residual data

from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

##dataset
df = pd.read_excel("D:/data science assignment/Assignments/18.Forecasting/Airlines+Data.xlsx",header=0,parse_dates=True)
df


#EDA
df.info()
df.describe()

df.set_index('Month',inplace=True)## making the month column as index
df

df.index.year

df.isnull().sum().sum()# no nan values in the data set

df[df.duplicated()].shape
#found the 16 duplicated rows

df[df.duplicated()]

df.drop_duplicates(inplace=True)
## removed the duplicated
df

DF = df.copy()
## copied the data from df to DF so that any changes done to coppied data does not reflect to original data
DF
DF.info()

DF.ndim
2
DF.isnull().sum().sum()
0
## Visualization of Data Checking the line plot,Histogram and Density Plots,create a density plot,
## Box and Whisker Plots by Interval,Lag Plot, Autocorrelation Plot
DF.plot()
plt.show()

## here we can say that the trend is upward and the sessionality is multiplicative
##Histogram and Density Plots

DF.hist()
plt.show()

# create a density plot
DF.plot(kind='kde')
plt.show()

##Lag_plot
lag_plot(DF)
plt.show()

#Autocorrelation Plot

plot_acf(DF,lags=30)
plt.show()

#UpSampling
upsampled = DF.resample('M').mean()
print(upsampled.head(32))

interpolated = upsampled.interpolate(method='linear') ## interplation was done for nan values which we get after doing upsampling by month
print(interpolated.head(15))
interpolated.plot()
plt.show()

Tranformations
# line plot
plt.subplot(211)
plt.plot(DF)


# histogram
plt.subplot(212)
plt.hist(DF)
plt.show()

#Square Root Transform
dataframe = DataFrame(DF.values)
dataframe.columns = ['Passengers']
dataframe['Passengers'] = sqrt(dataframe['Passengers'])
# line plot
plt.subplot(211)
plt.plot(DF['Passengers'])
# histogram
plt.subplot(212)
plt.hist(DF['Passengers'])
plt.show()

#Log Transform
from numpy import log
## importing the log library
dataframe = DataFrame(DF.values)
dataframe.columns = ['Passengers']
dataframe['Passengers'] = log(dataframe['Passengers'])

# line plot
plt.subplot(211)
plt.plot(dataframe['Passengers'])
# histogram
plt.subplot(212)
plt.hist(dataframe['Passengers'])
plt.show()

Train = interpolated.head(81)
Test = interpolated.tail(14)

#Moving Average
plt.figure(figsize=(12,4))
interpolated.Passengers.plot(label="org")
for i in range(2,24,6):
    interpolated["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')


#Time series decomposition plot
decompose_ts_add = seasonal_decompose(interpolated.Passengers,freq=12)  
decompose_ts_add.plot()
plt.show()


ACF plots and PACF plots
import statsmodels.graphics.tsaplots as tsa_plots

tsa_plots.plot_acf(interpolated.Passengers,lags=14)
tsa_plots.plot_pacf(interpolated.Passengers,lags=14)
plt.show()


Evaluation Metric MAPE
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)
#Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Passengers"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Passengers)

# Holt method 
hw_model = Holt(Train["Passengers"]).fit(smoothing_level=0.1, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Passengers)

#Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12).fit(smoothing_level=0.1, smoothing_slope=0.2) #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Passengers)

#Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit(smoothing_level=0.1, smoothing_slope=0.2) 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Passengers)

rmse_hwe_mul_add = sqrt(mean_squared_error(pred_hwe_mul_add,Test.Passengers))
rmse_hwe_mul_add

#Final Model by combining train and test
hwe_model_add_add = ExponentialSmoothing(interpolated["Passengers"],seasonal="add",trend="add",seasonal_periods=10).fit()

#Forecasting for next 10 time periods
hwe_model_add_add.forecast(10)

interpolated


interpolated.reset_index(inplace=True)
interpolated['t'] = 1

for i,row in interpolated.iterrows():
  interpolated['t'].iloc[i] = i+1

A value is trying to be set on a copy of a slice from a DataFrame




interpolated['t_sq'] = (interpolated['t'])**2
## inserted t_sq column with values
interpolated


interpolated["month"] = interpolated.Month.dt.strftime("%b") # month extraction
interpolated["year"] = interpolated.Month.dt.strftime("%Y") # month extraction
interpolated


months = pd.get_dummies(interpolated['month']) ## converting the dummy variables for month column
months


months = months[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]
## storing the months as serial wise again in months variable
Airlines = pd.concat([interpolated,months],axis=1)
Airlines.head()

Airlines['log_passengers'] = np.log(Airlines['Passengers'])

plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=Airlines,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")


# Boxplot 
plt.figure(figsize=(8,6))
plt.subplot(211)
sns.boxplot(x="month",y="Passengers",data= Airlines)
plt.subplot(212)
sns.boxplot(x="year",y="Passengers",data=Airlines)


plt.figure(figsize=(12,3))
sns.lineplot(x="year",y="Passengers",data=Airlines)


Splitting data
Train = Airlines.head(81) # training data
Test = Airlines.tail(14) # test Data


#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear
47.87107195088721
#Exponential
Exp = smf.ols('log_passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp
42.37179623821831

#Quadratic 
Quad = smf.ols('Passengers~t+t_sq',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_sq"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad
42.70987042515201

#Additive seasonality 

add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea
130.557623886014

#Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('Passengers~t+t_sq+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_sq']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad
26.78537191152376


##Multiplicative Seasonality

Mul_sea = smf.ols('log_passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
137.28596175917104

#Multiplicative Additive Seasonality 

Mul_Add_sea = smf.ols('log_passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea
13.188070730263183


#Compareing the results 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])

#rmse_multi_add_sea will be prefered than any other in this analysis
 


