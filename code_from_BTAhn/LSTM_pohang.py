# univariate data preparation

from numpy import array
# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
import sys


# enter the major paramteres via keyboard
n_steps=int(input('Enter n_steps(ex. 100):'))
n_neurons=int(input('Enter n_neurons(ex. 50):'))
n_epochs=int(input('Enter n_epochs(ex. 2000):'))   



df=pd.read_csv("/home/jspark/Python_Doc/Telco/code_from_BTAhn/2020-04-02_monitoring_data_crop1.csv", index_col="MTime", encoding='UTF-8')
# df=pd.read_csv("2020-04-02_monitoring_data.csv", index_col="MTime", encoding='UTF-8')
Avg_df = df.loc[:,["MXSineAvg","MYSineAvg"]]

ax = Avg_df.plot(title='Monitoring Data', figsize=(12,4),legend=True, fontsize=12)
ax.set_ylim([-0.1,0.1])
ax.set_xlabel('Time',fontsize=12)
ax.set_ylabel('Tilt', fontsize=12)
ax.grid(True)

ax.legend(['Tilt_x','Tilt_y'], fontsize=12)
plt.tight_layout()
for label in ax.xaxis.get_ticklabels() :
    label.set_rotation(45)



df1=pd.read_csv("/home/jspark/Python_Doc/Telco/code_from_BTAhn/pohang_temp_wind.csv", index_col="TIME", encoding='CP949')
pohang_df = df1.loc[:,["풍속(m/s)","기온(°C)"]]

ax0 = pohang_df.plot(figsize=(12,4),legend=True, fontsize=12)
ax0.set_title("Wind Speed and Temperature of POHANG")
ax0.set_ylabel("Wind Speed(m/s), Temperature(°C)")
ax0.set_xlabel("Time")
ax0.grid(True)

plt.tight_layout()
for label in ax0.xaxis.get_ticklabels() :
    label.set_rotation(45)



data_xsin = np.array(list(Avg_df['MXSineAvg']))
data_ysin = np.array(list(Avg_df['MYSineAvg']))
data_xsin_test = data_xsin
data_ysin_test = data_ysin



# create a index ranging from 1 to Num. of row
#print("Num. of Row ",dataset.shape)
n_min = 1
n_max = data_xsin.shape[0]
data_points_x = np.linspace(n_min, n_max, n_max)
data_points_y = np.linspace(n_min, n_max, n_max)

data_xsin = data_xsin.reshape(-1,1) # necessary for scaler fit_transform function
data_ysin = data_ysin.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
data_xsin = scaler.fit_transform(data_xsin) #scaling
data_ysin = scaler.fit_transform(data_ysin) #scaling



#n_steps = 200 # 
n_inputs = 1 # one input per time step
#n_neurons = 100 # hiddeln layer
n_outputs =  1 # output layer
n_features = 1
learning_rate = 0.0001
np.random.seed(1)

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# define input sequence
raw_seq_x = np.array(data_xsin.tolist())
raw_seq_y = np.array(data_ysin.tolist())



# split into samples
X_xsin, y_xsin = split_sequence(raw_seq_x, n_steps)
X_ysin, y_ysin = split_sequence(raw_seq_y, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]
X_xsin = X_xsin.reshape((X_xsin.shape[0], X_xsin.shape[1], n_features))
X_ysin = X_ysin.reshape((X_ysin.shape[0], X_ysin.shape[1], n_features))

# define model
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model = Sequential()

# # 2 layers
# model.add(LSTM(n_neurons, activation='relu', input_shape=(n_steps, n_features), return_sequences=True))
# model.add(LSTM(n_steps))


#=====================================
model.add(LSTM(n_neurons, activation='relu', input_shape=(n_steps, n_features)))
# model.add(Dense(5))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#====================================


# #====================================
# def model_fn():
#     # n_steps
#     # n_neurons
#     # n_epochs
    
#     n_steps=int(input('Enter n_steps(ex. 100):'))
#     n_neurons=int(input('Enter n_neurons(ex. 256):'))
#     LR = Choice('learning_rate', [0.001, 0.0005, 0.0001], group='optimizer')
#     DROPOUT_RATE = Linear('dropout_rate', 0.0, 0.5, 5, group='dense')
#     NUM_DIMS = Range('num_dims', 8, 32, 8, group='dense')
#     NUM_LAYERS = Range('num_layers', 1, 3, group='dense')
    
#     model.add(LSTM(n_neurons, activation='relu', input_shape=(n_steps, n_features)))
#     for _ in range(NUM_LAYERS):
#         model.add(Dense(NUM_DIMS, activation='relu'))
#         model.add(Dropout(DROPOUT_RATE))
#     model.add(Dense(1))
#     model.compile(optimizer=Adam(LR), loss='mse', metrics=['accuracy'])
    
#     return model

# tuner = Tuner(model_fn, 'val_accuracy' epoch_budget=500, max_epochs=5)
# tuner.search(X_xsin,validation_data=X_xsin)
#====================================



# fit model
print('####################################################')
print('               LSTM Training started!                ')
history_x = model.fit(X_xsin, y_xsin, epochs=n_epochs, batch_size = X_xsin.shape[0]//10, verbose=2, callbacks=[early_stopping])
# history_y = model.fit(X_ysin, y_ysin, epochs=n_epochs, batch_size = X_ysin.shape[0], verbose=0)
print('####################################################')
print('               Training completed!                ')
print(' ')

# evaluate
loss_x, accuracy_x = model.evaluate(X_xsin, y_xsin, verbose=0)
# loss_y, accuracy_y = model.evaluate(X_ysin, y_ysin, verbose=0)
print("loss_x ", loss_x, "accuracy_x ", accuracy_x)
# print("loss_y ", loss_y, "accuracy_y ", accuracy_y)



# demonstrate prediction
yhat_x = model.predict(X_xsin, verbose=0)
yhat_x = scaler.inverse_transform(yhat_x)
# yhat_y = model.predict(X_ysin, verbose=0)
# yhat_y = scaler.inverse_transform(yhat_y)

predicted_x = data_xsin_test
# predicted_y = data_ysin_test

raw_seq_x_scale = scaler.inverse_transform(raw_seq_x)


    
# Compare the predicted data to the original ones over given data
for i in range(0,yhat_x.shape[0]):
    predicted_x[n_steps+i] = yhat_x[i]


# graph1
plt.figure(figsize = (12,4))
# plt.ylim(-0.1, 0.6)
# plt.ylim(-0.1, 0.1)

plt1 = plt.plot(data_points_x, raw_seq_x_scale, label = 'Orginal')
plt.plot(data_points_x, predicted_x, label = 'Predicted')
plt.setp(plt1, linewidth=0.5)

plt.tight_layout()
# for label in ax.xaxis.get_ticklabels() :
#     label.set_rotation(45)

plt.title('Predicted Tilt Angles')
plt.xlabel('Time')
plt.ylabel('Tilt Angle')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


# graph2
plt.figure(figsize = (12,4))
plt.ylim(-0.015, 0.01)
plt.xlim(1500, 2000)

plt1 = plt.plot(data_points_x, raw_seq_x_scale, label = 'Orginal')
plt.plot(data_points_x, predicted_x, label = 'Predicted')
plt.setp(plt1, linewidth=0.5)

plt.tight_layout()
# for label in ax.xaxis.get_ticklabels() :
#     label.set_rotation(45)

plt.title('Predicted Tilt Angles')
plt.xlabel('Time')
plt.ylabel('Tilt Angle')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


# graph3
plt.figure(figsize = (12,4))
plt.ylim(-0.02, 0.01)
plt.xlim(1500, 1600)

plt1 = plt.plot(data_points_x, raw_seq_x_scale, label = 'Orginal')
plt.plot(data_points_x, predicted_x, label = 'Predicted')
plt.setp(plt1, linewidth=0.5)

plt.tight_layout()
# for label in ax.xaxis.get_ticklabels() :
#     label.set_rotation(45)

plt.title('Predicted Tilt Angles')
plt.xlabel('Time')
plt.ylabel('Tilt Angle')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# # Predict one more cycle
# num_row = Avg_df.shape[0]
# num_col = Avg_df.shape[1]

# x_input_xsin = X_xsin[X_xsin.shape[0]-1, 0:n_steps,n_features-1]
# x_input_xsin = x_input_xsin.reshape((1, n_steps, n_features))

# y_predict_xsin = np.arange(num_row)

# for i in range(0, num_row):
#     y_out_norm_xsin = model.predict(x_input_xsin, verbose=0)
#     back_x_input_xsin = x_input_xsin    
#     y_predict_xsin[i] = scaler.inverse_transform(y_out_norm_xsin)

#     for j in range(1,n_steps):
#         x_input_xsin[0,j-1,n_features-1] = back_x_input_xsin[0,j,n_features-1]
        
#     x_input_xsin[0,n_steps-1,n_features-1] = y_out_norm_xsin

# Avg_df['predicted'] = y_predict_xsin
# Avg_df.plot()
# hyper_par = 'n_steps='+str(n_steps)+' n_neurons='+str(n_neurons)+' epochs='+str(n_epochs)
# plt.ylabel("Tilt Angle")
# plt.title(hyper_par)

# plt.show()














# # file defining data and paramteres to be used in LSTM
# set_file = open("input_set.txt", "r")
# set_file.readline()
# read1 = set_file.readline()
# data_file = read1.strip()
# print("Data file name: ", data_file)
# #read2 = set_file.readline()
# #print("n_steps: ", read2)
# #n_steps = int( read2)
# #read3 = set_file.readline()
# #print("n_neurons: ", read3)
# #n_neurons = int( read3)
# #read4 = set_file.readline()
# #print("epochs: ", read4)
# #n_epochs = int(read4)      
# set_file.close()

# # enter the major paramteres via keyboard
# n_steps=int(input('Enter n_steps(ex. 100):'))
# n_neurons=int(input('Enter n_neurons(ex. 50):'))
# n_epochs=int(input('Enter n_epochs(ex. 2000):'))               

# # read data in the pandas dataframe format
# df = pd.read_excel(data_file)
# df.set_index("Age", inplace=True)
# df.plot()
# plt.ylabel("Frequency")
# plt.title("Traffic accidents")
# plt.show()

# # combine 5-year's data into a vector to used in creating samples

# num_row = df.shape[0]
# num_col = df.shape[1]

# WS = list(range(num_row*num_col))

# for i in range(0,num_row*num_col):
#     WS[i] = 0

# for j in range(0,num_col):
#     for i in range(0,num_row):
#         WS[j*num_row+i] = df.iloc[i,j]

# # convert a list into numpy array for use later
# dataset = np.array(WS)

# plt.plot(dataset)
# plt.ylabel("Frequency")
# plt.xlabel("Age")
# plt.title("Combined traffic accident")
# plt.show()

# # back up for later use
# backup_dataset = dataset

# # create a index ranging from 1 to Num. of row
# #print("Num. of Row ",dataset.shape)
# minimum = 1
# maximum = dataset.shape[0]
# data_points = np.linspace(minimum, maximum, maximum)


# dataset = dataset.reshape(-1,1) # necessary for scaler fit_transform function
# scaler = MinMaxScaler(feature_range=(0,1))
# dataset = scaler.fit_transform(dataset) #scaling


# #n_steps = 200 # 
# n_inputs = 1 # one input per time step
# #n_neurons = 100 # hiddeln layer
# n_outputs =  1 # output layer
# n_features = 1
# learning_rate = 0.0001
# np.random.seed(1)

# # split a univariate sequence into samples
# def split_sequence(sequence, n_steps):
#     X, y = list(), list()
#     for i in range(len(sequence)):
#     # find the end of this pattern
#         end_ix = i + n_steps
#         # check if we are beyond the sequence
#         if end_ix > len(sequence)-1:
#             break
#         # gather input and output parts of the pattern
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
#         X.append(seq_x)
#         y.append(seq_y)
#     return array(X), array(y)

# # define input sequence
# raw_seq = dataset.tolist()


# # split into samples
# X, y = split_sequence(raw_seq, n_steps)
# # summarize the data
# #for i in range(len(X)):
# #    print(X[i], y[i])

# # reshape from [samples, timesteps] into [samples, timesteps, features]

# X = X.reshape((X.shape[0], X.shape[1], n_features))
# # define model
# model = Sequential()
# model.add(LSTM(n_neurons, activation='relu', input_shape=(n_steps, n_features)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
# # fit model
# print('####################################################')
# print('               LSTM Training started!                ')
# history=model.fit(X, y, epochs=n_epochs, batch_size = X.shape[0], verbose=0)
# # evaluate
# print('####################################################')
# print('               Training completed!                ')
# print(' ')
# loss, accuracy = model.evaluate(X, y, verbose=0)
# print("loss ", loss, "accuracy ", accuracy)

# #from pydub import AudioSegment
# #from pydub.playback import play
# #song = AudioSegment.from_wav("Ring09.wav")
# #play(song)

# # demonstrate prediction
# yhat = model.predict(X, verbose=0)
# yhat = scaler.inverse_transform(yhat)

# predicted = backup_dataset
    
# # Compare the predicted data to the original ones over given data
# for i in range(0,yhat.shape[0]):
#     predicted[n_steps+i] = yhat[i]

# plt.plot(data_points, predicted, label = 'Predicted')
# raw_seq = scaler.inverse_transform(raw_seq)
# plt.plot(data_points, raw_seq, label = 'Orginal')
# plt.legend('Predicted traffic accident frequency')
# plt.xlabel('Age over year')
# plt.ylabel('Frequency')
# plt.legend(loc='upper right')
# plt.show()

# # Predict one more cycle
# x_input = X[X.shape[0]-1, 0:n_steps,n_features-1]
# x_input = x_input.reshape((1, n_steps, n_features))

# y_predict = np.arange(num_row)

# for i in range(0, num_row):
#     y_out_norm = model.predict(x_input, verbose=0)
#     back_x_input = x_input    
#     y_predict[i] = scaler.inverse_transform(y_out_norm)

#     for j in range(1,n_steps):
#         x_input[0,j-1,n_features-1] = back_x_input[0,j,n_features-1]
        
#     x_input[0,n_steps-1,n_features-1] = y_out_norm

# df['2019(predicted)'] = y_predict
# df.plot()
# hyper_par = 'n_steps='+str(n_steps)+' n_neurons='+str(n_neurons)+' epochs='+str(n_epochs)
# plt.ylabel("Frequency")
# plt.title(hyper_par)

# plt.show()



