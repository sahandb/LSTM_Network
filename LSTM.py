import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


def preProcess(path):
    global blocks
    with open(path) as f:
        lines = f.readlines()
    allLines = np.array(lines)

    allLinesSep = np.char.split(allLines, sep=' ')

    chunks = list()
    x = 0
    chunks.append([])  # create an empty chunk to which we'd append in the loop
    for i in allLinesSep:
        if i != ['', '', '', '', '', '', '', '', '', '', '', '', '\n']:
            chunks[x].append(i)
        else:
            x += 1
            chunks.append([])

    chunks = np.delete(chunks, 0)
    chunks = np.array(chunks).reshape((chunks.shape[0], 1))

    data = list()
    i = 0
    if path == "Data/Train_Arabic_Digit.txt":
        blocks = 660
    elif path == "Data/Test_Arabic_Digit.txt":
        blocks = 220
    for x in range(0, chunks.shape[0], blocks):
        for y in range(blocks):
            data.append((np.array(chunks[y]), i))
        i += 1
    data = np.array(data)
    return data


# asd = preProcess("Data/Train_Arabic_Digit.txt")
# asdd = preProcess("Data/Test_Arabic_Digit.txt")


with open("Data/Train_Arabic_Digit.txt") as f:
    lines = f.readlines()
allLines = np.array(lines)

allLinesSep = np.char.split(allLines, sep=' ')

chunks = list()
x = 0
chunks.append([])  # create an empty chunk to which we'd append in the loop
for i in allLinesSep:
    if i != ['', '', '', '', '', '', '', '', '', '', '', '', '\n']:
        chunks[x].append(i)
    else:
        x += 1
        chunks.append([])

chunks = np.delete(chunks, 0)
chunks = np.array(chunks).reshape((chunks.shape[0], 1))

data = list()
i = 0

for x in range(0, chunks.shape[0], 660):
    for y in range(660):
        data.append((np.array(chunks[y]), i))
    i += 1
data = np.array(data)

# bbbb = np.argsort(data[:,0])
sortedd = np.argsort(data[:, 0][:][::-1])
sortedData = list()
for i in sortedd:
    sortedData.append(data[i])

sortedData = np.array(sortedData)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(6600, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# train model
model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=100, batch_size=64, verbose=1)

# predicton
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# rmse for train an test
math.sqrt(mean_squared_error(y_train, train_predict))

math.sqrt(mean_squared_error(ytest, test_predict))

# Plotting
# shift train predictions for plotting
look_back = 100
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(data) - 1, :] = test_predict
# plot baseline and predictions
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
