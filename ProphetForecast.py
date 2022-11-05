#install prophet and yfinance
#in command line: python - m install prophet
#in command line: pip install yfinance
#can use plotly to make interactive plots, install plotly 4.0 or above seperately
#also need to install notebook and ipywidgets packages
from datetime import date
import yfinance as yf
from prophet import Prophet


#Get Data
START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

#Select Stock to analyze
selected_stock = "META"

data = yf.download(selected_stock, START, TODAY)
data.reset_index(inplace = True)

print(data.tail())

#Predict Forecast with Prophet

df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns ={"Date":"ds", "Close":"y"})

m = Prophet()
m.fit(df_train)

#Create Future Period to Predict
n_years = 1
period = n_years * 365
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)

forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

#plot forecast
figure1 = m.plot(forecast)

#plot trend, weekly, yearly forecast

figure2 = m.plot_components(forecast)

figure1.show()
figure2.show()

#Input to close, otherwise graphs disappear instantly
input("Press y to close: ")
