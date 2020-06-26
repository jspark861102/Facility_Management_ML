import datetime
import copy

from numpy import array
import pandas as pd

from keras.models import Sequential
from keras.layers import LSTM, Dense

def split_sequence(sequence, n_steps):
    X, y = list(), list()

    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    return array(X), array(y)

df = pd.read_csv('2020-04-02 모니터링 데이터.csv')
df = df[df.GHID == 5].copy()[['MXSineMax'] + ['MTime']]
df['MTime'] = pd.to_datetime(df.MTime)
df = df.set_index(['MTime'])
# print(df.head())

raw_seq = list(df.iloc[:, 0])
n_step = 3

X, y = split_sequence(raw_seq, n_step)

# for i in range(len(X)):
#     print(X[i], y[i])

n_feature = 1
X = X.reshape((X.shape[0], X.shape[1], n_feature))
# print(X)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_step, n_feature)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=200, verbose=0)

# 입력데이터셋을 1일당 1개 값(평균으로 시작)
# x_input = array()
# x_input = x_input.reshape(1, n_step, n_feature)

# y_hat = model.predict(x_input, verbose=0)
# print(y_hat)