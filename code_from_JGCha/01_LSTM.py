from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

## Data Preparation
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


raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n_step = 3

X, y = split_sequence(raw_seq, n_step)

# for i in range(len(X)):
#     print(X[i], y[i])

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

## Vanilla LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_step, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=200, verbose=0)
x_input = array([70, 80, 90])
x_input = x_input.reshape(1, n_step, n_features)

y_hat = model.predict(x_input, verbose=0)
print(y_hat)
