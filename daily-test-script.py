import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime, pickle
np.set_printoptions(suppress=True)

print(tf.__version__)
# This code allows for the GPU to be utilized properly.
tf.autograph.set_verbosity(0)
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

print(physical_devices)

def sampleDataset(data, num_samples, interval_len, feature_size, offset=0):
    # creates samples of data from a dataset
    # returns the intervals, and the target price changes

    samples = np.ndarray((num_samples, interval_len, feature_size))
    target_prices = np.ndarray((num_samples, feature_size))
    for x in range(num_samples):
        start_index = round(x * (data.shape[0] - interval_len - 1 - offset) / num_samples) + offset
        # data_range = data.shape[0] - offset

        samples[x] = data[start_index:start_index + interval_len]
        target_prices[x] = data[start_index + interval_len]
    
    return samples, target_prices

def normalize_data(data, method="one"):
    if method=="one":
        maxes = np.max(data, axis=0)
        mins = np.min(data, axis=0)
        return (data - mins) / (maxes - mins)
    elif method=="zscore":
        mean = np.mean(data, axis=0)
        stdev = np.std(data, axis=0)
        return (data - mean) / stdev


# === HYPERPARAMETERS === #
DATASET_SIZE = 10092 # Number of samples PER TICKER
TICKER = ["^GSPC"]
INTERVAL = 7
FEATURE_SIZE = 5 # open, high, low, close, volume
DATA_OFFSET = 14000 # offset from latest datanum_epochs = 1000

num_epochs = 500
batch_size = 256
LEARNING_RATE = 0.001
REG_FACTOR = 0
VAL_RATIO = 0



# === GET TICKER DATA FOR TODAY === #
hists = []
for x in range(len(TICKER)):
    print("LOADING TICKER " + str(x) + "/" + str(len(TICKER)), end="\r")
    hist = yf.Ticker(TICKER[x]).history(interval="1d", period="max")
    if not hist.empty and np.sum(np.isnan(hist.loc[:, "Open"].to_numpy())) == 0:
        hist.drop(columns=["Dividends", "Stock Splits"], inplace=True)
        hists.append(hist)

price_intervals = np.ndarray((len(hists), DATASET_SIZE, INTERVAL, FEATURE_SIZE))
target_prices = np.ndarray((len(hists), DATASET_SIZE, FEATURE_SIZE))
max_prices = np.ndarray((len(hists),))
min_prices = np.ndarray((len(hists),))
means = np.ndarray((len(hists),))
stdevs = np.ndarray((len(hists),))

# Process features/data
for x in range(len(hists)):

    hist = hists[x]
    columns = [hist.loc[:, col].to_numpy() for col in hist.columns]

    max_prices[x] = np.max(columns[0])
    min_prices[x] = np.min(columns[0])
    means[x] = np.mean(columns[0])
    stdevs[x] = np.std(columns[0])
    data = np.stack(columns, axis=1)
    data = normalize_data(data, method="zscore")

    price_intervals[x], target_prices[x] = sampleDataset(data, DATASET_SIZE, INTERVAL, FEATURE_SIZE, offset=DATA_OFFSET)


data_split = np.copy(price_intervals)
targets_split = np.copy(target_prices)

price_intervals = price_intervals.reshape((len(hists) * DATASET_SIZE, INTERVAL, FEATURE_SIZE))
target_prices = target_prices.reshape((len(hists) * DATASET_SIZE, FEATURE_SIZE))



# === MAKE AND TRAIN MODEL === #

# Define the LSTM model
model = keras.Sequential()
model.add(keras.layers.Input((INTERVAL, FEATURE_SIZE)))
model.add(keras.layers.LSTM(units=64, kernel_regularizer=keras.regularizers.l2(REG_FACTOR), return_sequences=True))
model.add(keras.layers.LSTM(units=32, kernel_regularizer=keras.regularizers.l2(REG_FACTOR)))
model.add(keras.layers.Dense(units=64, activation="relu"))
# model.add(keras.layers.Dense(units=64, activation="relu"))
model.add(keras.layers.Dense(units=FEATURE_SIZE, activation="linear"))

# Compile the model
model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

model.build((None, INTERVAL, FEATURE_SIZE))
model.summary()

callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

# Train the model
train_hist = model.fit(price_intervals, target_prices, epochs=num_epochs, batch_size=batch_size, validation_split=VAL_RATIO, callbacks=[])
model.save("DAILY_PREDICTION_MODEL")

# Evaluate the model
metrics = model.evaluate(price_intervals, target_prices)

# Predict on latest
predictions = model.predict(np.expand_dims(price_intervals[-1], axis=0))



# === DISPLAY DATA === #
print()
print("=== PREDICTION FOR NEXT TRADING DAY: ===")
print("Today's date: " + datetime.datetime.now().strftime("%A, %B %d"))
print("Predicted next open price:  " + str(predictions[0, 0] * stdevs[0] + means[0]))
# print("Predicted next high price:  " + str(predictions[0, 1]))
# print("Predicted next low price:   " + str(predictions[0, 2]))
# print("Predicted next close price: " + str(predictions[0, 3]))
# print("Predicted next volume:      " + str(predictions[0, 4]))