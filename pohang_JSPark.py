# univariate data preparation

from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import sys

#learning parameters
n_steps = 200 # 
n_neurons = 512 # hiddeln layer
n_epoch = 500
n_outputs =  1 # output layer
n_features = 1

########################################
#load data
df=pd.read_csv("/home/jspark/Python_Doc/Telco/2020-04-02_monitoring_data.csv", encoding='UTF-8', )
Avg_df = df.loc[:,["MYSineAvg"]]
raw = np.array(list(Avg_df['MYSineAvg']))
raw = raw[9:]
raw_x = np.linspace(1, raw.shape[0], raw.shape[0]) #x-axis

#plot
plt.figure(1,figsize = (12,4))
plt.plot(raw_x, raw, label = 'raw_data')
plt.legend(loc='upper right')
plt.grid(True)
#plt.show()

#split
split_number_0 = raw.shape[0]
split_number_1 = 27000
split_number_2 = 33900
split_number_3 = 34000

raw_train = raw[0:split_number_1] #some of initial values can not be used
raw_test1 = raw[split_number_1:]                #both
raw_test2 = raw[split_number_1:split_number_2]  #only before
raw_test3 = raw[split_number_3:]                #only after

train_x = np.linspace(0, split_number_1, split_number_1 - 0)
test1_x = np.linspace(split_number_1, split_number_0, split_number_0 - split_number_1)
test2_x = np.linspace(split_number_1, split_number_2, split_number_2 - split_number_1)
test3_x = np.linspace(split_number_3, split_number_0, split_number_0 - split_number_3)

#plot
plt.figure(2,figsize = (12,4))
plt.plot(raw_x, raw, label = 'raw_data')
plt.plot(train_x, raw_train, label = 'train')
plt.plot(test1_x, raw_test1, label = 'test1')
plt.plot(test2_x, raw_test2, label = 'test2')
plt.plot(test3_x, raw_test3, label = 'test3')
plt.legend(loc='upper right')
plt.grid(True)
#plt.show()
########################################

########################################
#scaling
raw_train_reshape = raw_train.reshape(-1,1) # necessary for scaler fit_transform function
raw_test1_reshape = raw_test1.reshape(-1,1) # necessary for scaler fit_transform function
raw_test2_reshape = raw_test2.reshape(-1,1) # necessary for scaler fit_transform function
raw_test3_reshape = raw_test3.reshape(-1,1) # necessary for scaler fit_transform function

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(raw_train_reshape)

train_scaled = scaler.transform(raw_train_reshape) #scaling
test1_scaled = scaler.transform(raw_test1_reshape) #scaling
test2_scaled = scaler.transform(raw_test2_reshape) #scaling
test3_scaled = scaler.transform(raw_test3_reshape) #scaling

train_scaled_rescale = scaler.inverse_transform(train_scaled)
test1_scaled_rescale = scaler.inverse_transform(test1_scaled)
test2_scaled_rescale = scaler.inverse_transform(test2_scaled)
test3_scaled_rescale = scaler.inverse_transform(test3_scaled)
########################################


########################################
#data split
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)

# define input sequence
train_scaled_list = np.array(train_scaled.tolist())
test1_scaled_list = np.array(test1_scaled.tolist())
test2_scaled_list = np.array(test2_scaled.tolist())
test3_scaled_list = np.array(test3_scaled.tolist())

# split into samples
train_scaled_list_x, train_scaled_list_y = split_sequence(train_scaled_list, n_steps)
test1_scaled_list_x, test1_scaled_list_y = split_sequence(test1_scaled_list, n_steps)
test2_scaled_list_x, test2_scaled_list_y = split_sequence(test2_scaled_list, n_steps)
test3_scaled_list_x, test3_scaled_list_y = split_sequence(test3_scaled_list, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]
train_scaled_list_x_reshape = train_scaled_list_x.reshape((train_scaled_list_x.shape[0], train_scaled_list_x.shape[1], n_features))
test1_scaled_list_x_reshape = test1_scaled_list_x.reshape((test1_scaled_list_x.shape[0], test1_scaled_list_x.shape[1], n_features))
test2_scaled_list_x_reshape = test2_scaled_list_x.reshape((test2_scaled_list_x.shape[0], test2_scaled_list_x.shape[1], n_features))
test3_scaled_list_x_reshape = test3_scaled_list_x.reshape((test3_scaled_list_x.shape[0], test3_scaled_list_x.shape[1], n_features))
########################################


# define model
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model = Sequential()

#=====================================
model.add(LSTM(n_neurons, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#====================================

# fit model
print('####################################################')
print('               LSTM Training started!                ')
history_x = model.fit(train_scaled_list_x_reshape, train_scaled_list_y, epochs=n_epoch, batch_size = train_scaled_list_x_reshape.shape[0]//10, verbose=2, callbacks=[early_stopping])
print('####################################################')
print('               Training completed!                ')
print(' ')

# evaluate
loss_1, accuracy_1 = model.evaluate(test1_scaled_list_x_reshape, test1_scaled_list_y, verbose=0)
loss_2, accuracy_2 = model.evaluate(test2_scaled_list_x_reshape, test2_scaled_list_y, verbose=0)
loss_3, accuracy_3 = model.evaluate(test3_scaled_list_x_reshape, test3_scaled_list_y, verbose=0)
print("loss_1 ", loss_1, "accuracy_1 ", accuracy_1)
print("loss_2 ", loss_2, "accuracy_2 ", accuracy_2)
print("loss_3 ", loss_3, "accuracy_3 ", accuracy_3)

# demonstrate prediction
y_pred1 = model.predict(test1_scaled_list_x_reshape, verbose=0)
y_pred2 = model.predict(test2_scaled_list_x_reshape, verbose=0)
y_pred3 = model.predict(test3_scaled_list_x_reshape, verbose=0)
y_pred1_rescale = scaler.inverse_transform(y_pred1)
y_pred2_rescale = scaler.inverse_transform(y_pred2)
y_pred3_rescale = scaler.inverse_transform(y_pred3)

# demonstrate train
y_train = model.predict(train_scaled_list_x_reshape, verbose=0)
y_train_rescale = scaler.inverse_transform(y_train)

# create a index ranging from 1 to Num. of row
y_train_x = train_x[n_steps:]
y_pred1_x = test1_x[n_steps:]
y_pred2_x = test2_x[n_steps:]
y_pred3_x = test3_x[n_steps:]

#
scaled_list = np.concatenate([train_scaled_list, test1_scaled_list])
scaled_list_rescale = scaler.inverse_transform(scaled_list)

# graph
plt.figure(3,figsize = (12,4))
plt.plot(raw_x, scaled_list_rescale, label = 'Original')
plt.plot(y_train_x, y_train_rescale, label = 'train')
plt.plot(y_pred1_x, y_pred1_rescale, label = 'test1')
plt.tight_layout()
plt.title('Predicted Tilt Angles')
plt.xlabel('Time')
plt.ylabel('Tilt Angle')
plt.legend(loc='upper right')
plt.grid(True)
#plt.show()

plt.figure(4,figsize = (12,4))
plt.plot(raw_x, scaled_list_rescale, label = 'Original')
plt.plot(y_train_x, y_train_rescale, label = 'train')
plt.plot(y_pred2_x, y_pred2_rescale, label = 'test2')
plt.tight_layout()
plt.title('Predicted Tilt Angles')
plt.xlabel('Time')
plt.ylabel('Tilt Angle')
plt.legend(loc='upper right')
plt.grid(True)
#plt.show()

plt.figure(5,figsize = (12,4))
plt.plot(raw_x, scaled_list_rescale, label = 'Original')
plt.plot(y_train_x, y_train_rescale, label = 'train')
plt.plot(y_pred3_x, y_pred3_rescale, label = 'test3')
plt.tight_layout()
plt.title('Predicted Tilt Angles')
plt.xlabel('Time')
plt.ylabel('Tilt Angle')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


