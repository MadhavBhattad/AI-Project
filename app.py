from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
import streamlit as st

st.title('Stock Price Prediction')

user_input = st.text_input('Enter Stock Ticker', 'GOOGL')
df = yf.download(user_input)


#describing data
st.subheader('Data from the start')
st.write(df.describe())

#visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig) 

st.subheader('Closing Price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig) 

st.subheader('Closing Price vs Time chart with 200MA')
ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig) 

st.subheader('Closing Price vs Time chart with 100MA and 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig) 

#Splitting data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#Load the model
model = load_model('keras_model.h5')

#Testing part

past_100_days = pd.DataFrame(data_training.tail(100))
final_df = pd.concat([past_100_days,data_testing], ignore_index=True)
#final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
 x_test.append(input_data[i-100: i])
 y_test.append(input_data[i,0])
 
x_test, y_test = np.array(x_test), np.array(y_test)

#Make the prediction
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final graph
st.subheader('Predictions VS Original')
fig2, ax = plt.subplots(figsize=(12,6))
ax.plot(y_test, 'b', label = 'Original Price')
ax.plot(y_predicted, 'r', label = 'Predicted Price')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig2)
