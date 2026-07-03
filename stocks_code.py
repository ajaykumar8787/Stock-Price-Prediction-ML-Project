# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Step 2: Load Training Dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values  # 'Open' column

# Step 3: Visualization of Dataset
plt.figure(figsize=(10,6))
plt.plot(training_set, color='red', label='Google Stock Price (Train)')
plt.title('Google Stock Price History')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Step 4: Feature Scaling
scaler = MinMaxScaler(feature_range=(0,1))
scaled_training = scaler.fit_transform(training_set)

# Step 5: Preparing the Dataset for Training
X_train = []
y_train = []
for i in range(60, len(scaled_training)):
    X_train.append(scaled_training[i-60:i, 0])
    y_train.append(scaled_training[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Step 6: Reshaping the Dataset
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Step 7: Model Development
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Step 8: Compile & Train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Step 9: Load Test Dataset
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Step 10: Preprocessing Test Data
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, 60+len(dataset_test)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Step 11: Predicting the Output
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Step 12: Result Visualization
plt.figure(figsize=(10,6))
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
