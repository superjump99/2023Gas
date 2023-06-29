from Func_set import *
from EvalFunc import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader, TensorDataset # 데이터로더,텐서데이터셋


def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []

    for i in range(0, len(time_series)-seq_length):
        _x = time_series[i:i+seq_length, :]         # 전체 데이터의 0~ 시계열 길이까지
        _y = time_series[i+seq_length, 0:1]        # CH4 부분
        # print(_x, "-->",_y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

class Model(nn.Module):
    def __init__(self,target_size, hidden_size,num_layers, dropout):
        super(Model, self).__init__()
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.num_layer = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=7, hidden_size=self.hidden_size, num_layers=self.num_layer,
                            # bidirectional=True,옵티마이저
                            batch_first=True)     # True =>[batch_size, seq_length, input_size]

        self.fc1 = nn.Linear(in_features=seq_length * self.hidden_size, out_features=1)      # seq_length * self.hidden_size => 모델에서 (시퀀스 길이 * hidden_size)로 출력
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(in_features=1, out_features=self.target_size)


    def forward(self, x):
        # 초기 은닉상태        # h0 size [층의 개수(num_layers), 배치크기, 은닉size(hidden_size)]
        h0 = torch.zeros(num_layers, data.shape[0], hidden_size).to(device)
        C0 = torch.zeros(num_layers, data.shape[0], hidden_size).to(device)
        x, hn = self.lstm(x, (h0,C0))   # RNN층의 출력
        # h0 = 초기 은닉 상태
        x = torch.reshape(x, (x.shape[0],-1))
        # x.shape[0] -> batch_size
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = torch.flatten(x)
        return x
    # 평가지표


## 조정 가능한 하이퍼 파라미터
is_train_mode = True
# individual = True
feature_size = 7
target_size = 1
seq_length = 10
num_layers = 10
hidden_size = 10
mini_batch = 1
train_rate = 0.8
Epoch = 300
patience = 5
dropout = 0.2
learning_rate = 0.001
saved_model_name = 'LSTM.tar'

#=====================

if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    place = [i for i in range(700)]
    ran_place = random.sample(place, 100)
    random.shuffle(ran_place)
    testplace = random.sample(place, 1)

    # 학습
    if is_train_mode:

        for i in ran_place:
            # print(f"{i}번째 장소 선택 : {ran_place[i]}")
            df = pd.read_excel('../변환 데이터/TSD.xlsx', sheet_name=i)  # V2B-18
            df.set_index('Date Time', inplace=True)
            df = data_processing(df)
            df = df.fillna(df.mean()['CH4':'VOR'])

            # 학습 테스트 사이즈 구분

            train_size = int(len(df) * train_rate)
            train_set = df[0:train_size]
            test_set = df[train_size - seq_length:]
            real_y = list(test_set.iloc[seq_length:, 0])

            # Input scale
            scaler_x = MinMaxScaler()
            scaler_x.fit(train_set.iloc[:, 1:])

            train_set.iloc[:, 1:] = scaler_x.transform(train_set.iloc[:, 1:])
            test_set.iloc[:, 1:] = scaler_x.transform(test_set.iloc[:, 1:])

            # Output scale
            scaler_y = MinMaxScaler()
            scaler_y.fit(train_set.iloc[:, [0]])
            train_set.iloc[:, 0] = scaler_y.transform(train_set.iloc[:, [0]])
            test_set.iloc[:, 0] = scaler_y.transform(test_set.iloc[:, [0]])

            trainX, trainY = build_dataset(np.array(train_set), seq_length)
            testX, testY = build_dataset(np.array(test_set), seq_length)
            # 텐서로 변환
            trainX_tensor = torch.FloatTensor(trainX)
            trainY_tensor = torch.FloatTensor(trainY)

            testX_tensor = torch.FloatTensor(testX)
            testY_tensor = torch.FloatTensor(testY)

            # 텐서 형태로 데이터 정의
            Train_set = TensorDataset(trainX_tensor, trainY_tensor)
            Test_set = TensorDataset(testX_tensor, testY_tensor)

            # 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
            Train_data = DataLoader(Train_set, batch_size=mini_batch, shuffle=False, drop_last=True)  # 학습에 사용
            Test_data = DataLoader(Test_set, batch_size=mini_batch, shuffle=False, drop_last=True)  # 학습에 사용

            # ## 모델, optimization 설정
            model = Model(target_size=target_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(
                device)
            optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

            # 계속학습을 위한 모델 불러오기 (학습 중 이전에 학습시킨 모델 불러오기)
            if os.path.isfile(saved_model_name):
                # 불러오고 적용하기
                checkpoint = torch.load(saved_model_name)

                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                checkpoint_epoch = checkpoint['epoch']

            # 학습
            epoch = Epoch
            # epoch마다 loss 저장
            train_hist = np.zeros(epoch)
            for epoch in range(epoch):
                iterator = tqdm(Train_data, ascii= ' =')
                avg_cost = 0
                total_batch = len(Train_data)

                for data, label in iterator:
                    optimizer.zero_grad()

                    # 모델의 예측값
                    pred = model(data.type(torch.FloatTensor).to(device))

                    # 손실 계산
                    loss = nn.MSELoss()(pred, label.type(torch.FloatTensor).to(device))

                    loss.backward()         # 오차 역전파
                    optimizer.step()            # 최적화 진행
                    avg_cost += loss / total_batch

                    iterator.set_description(f'epoch{epoch}  loss:{loss.item()}')

                train_hist[epoch] = avg_cost
                # patience번째 마다 early stopping 여부 확인
                if (epoch % patience == 0) & (epoch != 0):
                    # loss가 커졌다면 early stop
                    if train_hist[epoch - patience] < train_hist[epoch]:
                        print('\n Early Stopping')
                        break

            # 예측 실행
            Test_data = DataLoader(Test_set, batch_size=1, shuffle=False, drop_last=True)
            predict = []
            total_loss = 0

            with torch.no_grad():
                # model.load_state_dict(torch.load(saved_model_name, map_location=device))

                for data, label in Test_data:

                    pred = model(data.type(torch.FloatTensor).to(device))

                    loss = nn.MSELoss()(pred, label.type(torch.FloatTensor).to(device))
                    total_loss += loss / len(Test_data)

                    pred = pred.cpu().numpy()
                    mean = pred.mean()
                    predict.append(mean)
            # print(type(predict[0]))

            label_y = list(test_set.iloc[seq_length:, 0])
            label_y = np.array(label_y)

            MSE =  mean_squared_error(predict, label_y)
            R2 = r2_score(label_y,predict)

            print('MSE SCORE : ', MSE)
            print('R2 SOCRE : ', R2)

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, saved_model_name)
    # 평가
    else:
        testplace = random.sample(place, 1)

        df = pd.read_excel('../변환 데이터/TSD.xlsx', sheet_name=testplace[0])  # V2B-18
        df.set_index('Date Time', inplace=True)
        df = data_processing(df)
        train_size = int(len(df) * train_rate)
        train_set = df[0:train_size]
        test_set = df[train_size - seq_length:]

        # Input scale
        scaler_x = MinMaxScaler()
        scaler_x.fit(train_set.iloc[:, 1:])

        train_set.iloc[:, 1:] = scaler_x.transform(train_set.iloc[:, 1:])
        test_set.iloc[:, 1:] = scaler_x.transform(test_set.iloc[:, 1:])

        # Output scale
        scaler_y = MinMaxScaler()
        scaler_y.fit(train_set.iloc[:, [0]])
        train_set.iloc[:, 0] = scaler_y.transform(train_set.iloc[:, [0]])
        test_set.iloc[:, 0] = scaler_y.transform(test_set.iloc[:, [0]])

        trainX, trainY = build_dataset(np.array(train_set), seq_length)
        testX, testY = build_dataset(np.array(test_set), seq_length)
        # 텐서로 변환
        trainX_tensor = torch.FloatTensor(trainX)
        trainY_tensor = torch.FloatTensor(trainY)

        testX_tensor = torch.FloatTensor(testX)
        testY_tensor = torch.FloatTensor(testY)

        # 텐서 형태로 데이터 정의
        Train_set = TensorDataset(trainX_tensor, trainY_tensor)
        Test_set = TensorDataset(testX_tensor, testY_tensor)

        # 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
        Train_data = DataLoader(Train_set, batch_size=mini_batch, shuffle=False, drop_last=True)  # 학습에 사용
        Test_data = DataLoader(Test_set, batch_size=mini_batch, shuffle=False, drop_last=True)  # 학습에 사용

        model = Model(target_size=target_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(
            device)
        optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

        checkpoint = torch.load(saved_model_name)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_epoch = checkpoint['epoch']

        predict = []
        total_loss = 0
        with torch.no_grad():

            for data, label in Test_set:

                pred = model(data.type(torch.FloatTensor).to(device))

                loss = nn.MSELoss()(pred, label.type(torch.FloatTensor).to(device))
                total_loss += loss / len(Test_set)

                pred = pred.cpu().numpy()
                mean = pred.mean()
                predict.append(mean)
        label_y = list(test_set.iloc[seq_length:, 0])
        label_y = np.array(label_y)

        MAE = MAE(predict, label_y)
        MSE = mean_squared_error(predict, label_y)
        RMSE = RMSE(predict, label_y)
        RMSLE = RMSLE(predict, label_y)
        MAPE = MAPE(predict, label_y)
        R2 = r2_score(label_y, predict)

        print('MSE SCORE : ', MSE)
        print('R2 SOCRE : ', R2)