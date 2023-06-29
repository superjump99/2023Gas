# 0. 필요한 라이브러리 import
from Func_set import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from time import time
import os

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# 현재 경로 확인
currentpath = os.getcwd()
print(currentpath)

# 데이터 불러오기
NTSD = non_time_series_data()
data = data_processing(NTSD)
print(data.head())
print(data.isnull().sum())

# 데이터 전처리
data = data.dropna()
print(data.info())


# target인 CH4를 제외한 X값과 target값인 CH4를 y로 분할
X = data.drop(['CH4'], axis=1)
y = data['CH4']

# 전체 데이터를 섞어서  train, test 분할
# Train data = 전체 0.7, test data = 전체 0.3


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Normalization 전과 후를 비교를 위한 scaling
# 학습 데이터 전처리
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_num_scaled = scaler.transform(X_train)

# 평가 데이터에 대한 전처리
X_test_prepared = scaler.transform(X_test)

# 2. 모델 구성하기
input_node = 6
hidden_node = [64, 64, 32]
output_node = 1

# Normalization 안한 모델
model = Sequential()
model.add(Dense(hidden_node[0], activation = 'relu', name = 'Hidden1',
                input_dim = input_node, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(hidden_node[1], activation = 'relu', name = 'Hidden2'))
model.add(Dense(hidden_node[2], activation = 'linear', name = 'Hidden3'))
model.add(Dense(output_node))

# Normalization 모델 (Min-max)
scale_model = Sequential()
scale_model.add(Dense(hidden_node[0], activation = 'relu', name = 'Hidden1',
                input_dim = input_node, kernel_initializer='random_uniform', bias_initializer='zeros'))
scale_model.add(Dense(hidden_node[1], activation = 'relu', name = 'Hidden2'))
scale_model.add(Dense(hidden_node[2], activation = 'linear', name = 'Hidden3'))
scale_model.add(Dense(output_node))

# 3. 모델 학습과정 설정하기

model.compile(loss = 'MSE', optimizer = 'adam', metrics = ['mae', 'mse'])

scale_model.compile(loss = 'MSE', optimizer = 'adam', metrics = ['mae', 'mse'])

# 4. 모델 학습

# Normalization 안한 모델
print("Start Training Non_normalization Model")
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(X_train, y_train, epochs = 28, batch_size = 10, validation_split = 0.3, callbacks=[tensorboard])

print("\n"*3,'Start Training Normalization Model')

# Normalization 모델 (Min-max)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
scale_model.fit(X_train_num_scaled, y_train, epochs = 28, batch_size = 10, validation_split = 0.3, callbacks=[tensorboard])

# 6. 모델 평가하기

# Normalization 안한 모델
result = model.evaluate(X_test, y_test, batch_size = 10)
print('Test Loss and MAE/MSE : ', result)
y_pred = model.predict(X_test)
print(y_pred)


# Normalization 모델 (Min-max)
scale_result = scale_model.evaluate(X_test_prepared, y_test, batch_size = 10)
print('Test Loss and MAE/MSE : ', scale_result)
y_pred_scale = scale_model.predict(X_test_prepared)
print(y_pred_scale)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Normalization 안한 모델
print("-----Model-----")
MAE = mean_absolute_error(y_test, y_pred)
RMSE = mean_squared_error(y_test, y_pred, squared=False)
MSE = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MAE :", MAE, "RMSE :", RMSE,  "MSE :", MSE, "r2 :", r2)

# Normalization 모델 (Min-max)
print("-----Scale_Model-----")
MAE = mean_absolute_error(y_test, y_pred_scale)
RMSE = mean_squared_error(y_test, y_pred_scale, squared=False)
MSE = mean_squared_error(y_test, y_pred_scale)
r2 = r2_score(y_test, y_pred_scale)
print("MAE :", MAE, "RMSE :", RMSE,  "MSE :", MSE, "r2 :", r2)