# Run: streamlit run web_bitcoin_price_predictor.py
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

st.title("BTC Price Prediction App")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-10,end.month,end.day)

bit_coin_data = yf.download("BTC-USD", start, end)
bit_coin_data.columns = bit_coin_data.columns.get_level_values(0)

model = load_model("Latest_bit_coin_model.keras")
st.subheader("Bitcoin Dataset (BTC-USD)")
st.write(bit_coin_data)

splitting_len = int(len(bit_coin_data)*0.9)
x_test = pd.DataFrame(bit_coin_data.Close[splitting_len:])

st.subheader('Original Close Price')
figsize = (15,6)
fig = plt.figure(figsize=figsize)
plt.plot(bit_coin_data.Close,'b')
st.pyplot(fig)

st.subheader("Test Close Price (Year-over-Year)")
st.write(x_test)

st.subheader('Test Close Price Chart (Year-over-Year)')
figsize = (15,6)
fig = plt.figure(figsize=figsize)
plt.plot(x_test,'b')
st.pyplot(fig)

# preprocess the data
# preprocessing the data into [0 to 1] range
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']].values)

x_data = []
y_data = []
for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)


ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = bit_coin_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot()

plt.plot(pd.concat([bit_coin_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data - Not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)

st.subheader("Future Price values")


last_100 = bit_coin_data[['Close']].tail(100)
last_100 = scaler.fit_transform(last_100['Close'].values.reshape(-1,1)).reshape(1,-1,1)

def predict_future(no_of_days, prev_100):
    future_predictions = []
    current_window = prev_100.copy()
    
    for _ in range(no_of_days):
        # next_day shape is likely (1, 1)
        next_day = model.predict(current_window)
        future_predictions.append(scaler.inverse_transform(next_day))
        
        # Append the prediction and slice out the first element to slide the window
        current_window = np.append(current_window[:, 1:, :], [next_day], axis=1)
        
    return future_predictions

no_of_days = st.number_input("Enter the nuymbers of days to be predicted from current date: (MAX: 30) ",7, 30, 7)
future_results = predict_future(no_of_days,last_100)
future_results = np.array(future_results).reshape(-1,1)

fig = plt.figure(figsize=(15, 6))
plt.plot(pd.DataFrame(future_results), marker = 'o')

for i in range(len(future_results)):
    y_val = float(future_results[i][0])
    plt.text(i, y_val, f"{int(y_val)}")
    
plt.xlabel('Days')
plt.ylabel('Close Price (USD)')

# GENERATE CUSTOM LABELS: ['Day 1', 'Day 2', 'Day 3', ...]
day_labels = [f'D-{i+1}' for i in range(no_of_days)]
plt.xticks(range(no_of_days), day_labels)

ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(style='plain', axis='y')

plt.title(f'Predicted Closing Price of BTC-USD for next {no_of_days} days')
st.pyplot(fig)

st.subheader("Summary")
st.write("- An **upward** trend may suggest that Bitcoin is likely to continue **rising**.")
st.write("- A **downward** trend may indicate that Bitcoin is likely to continue **falling**.")
st.write("- However, no trend guarantees future price movements.")