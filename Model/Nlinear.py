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

class LTSF_NLinear(torch.nn.Module):
    def __init__(self, window_size, forcast_size, individual, feature_size):
        super(LTSF_NLinear, self).__init__()
        self.window_size = window_size      # seq_length
        self.forcast_size = forcast_size       # target
        self.individual = individual
        self.channels = feature_size        # feature_size
        if self.individual:
            self.Linear = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(torch.nn.Linear(self.window_size, self.forcast_size))
        else:
            self.Linear = torch.nn.Linear(self.window_size, self.forcast_size)

    def forward(self, x):
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.forcast_size, x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x


## 조정 가능한 하이퍼 파라미터
is_train_mode = True
individual = True
feature_size = 7
target_size = 1
seq_length = 7

mini_batch = 4
train_rate = 0.8
Epoch = 100
patience = 10
learning_rate = 0.001
saved_model_name = 'Nlinear.tar'
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

        for i in range(699):
            # print(f"{i}번째 장소 선택 : {ran_place[i]}")
            print(f'{i}번째 Place')
            df = pd.read_excel('../변환 데이터/TSD.xlsx',sheet_name=i)  #V2B-18
            df.set_index('Date Time', inplace=True)
            df = data_processing(df)
            # df = df.loc[:,['CH4','P','T']]
            # print(df)
            # exit()
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
            model = LTSF_NLinear(window_size=seq_length,forcast_size=target_size,individual=individual,feature_size=feature_size).to(device)
            optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

            # 이전 학습 모델 불러와 다시 계속 학습
            if os.path.isfile(saved_model_name):
                # 불러오고 적용하기
                print('load model')
                checkpoint = torch.load(saved_model_name)

                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                checkpoint_epoch = checkpoint['epoch']

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
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, saved_model_name)

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
                pred_inverse = np.array(predict)
                # pred_inverse = scaler_y.inverse_transform(np.array(predict).reshape(-1, 1))
                # testY_inverse = scaler_y.inverse_transform((testY_tensor).reshape(-1, 1))
                # print(type(predict[0]))
                real_y = np.array(real_y)

            MSE = mean_squared_error(real_y, pred_inverse)
            MAPE = mean_absolute_percentage_error(real_y, pred_inverse)
            R2 = r2_score(real_y, pred_inverse)
            #
            print('MSE SCORE : ', MSE)
            print("MAPE SCORE : ", MAPE)
            print('R2 SOCRE : ', R2)

    # 평가
    else:
        df = pd.read_excel('../변환 데이터/TSD.xlsx', sheet_name=101)  # V2B-18
        df.set_index('Date Time', inplace=True)
        df = data_processing(df)
        # df = df.loc[:, ['CH4', 'P', 'T']]

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
        # Train_data = DataLoader(Train_set, batch_size=mini_batch, shuffle=False, drop_last=True)  # 학습에 사용

        # ## 모델, optimization 설정
        model = LTSF_NLinear(window_size=seq_length, forcast_size=target_size, individual=individual,
                             feature_size=feature_size).to(device)
        optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

        # 예측 실행
        Test_data = DataLoader(Test_set, batch_size=1, shuffle=False, drop_last=True)
        predict = []
        total_loss = 0

        checkpoint = torch.load(saved_model_name)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_epoch = checkpoint['epoch']

        with torch.no_grad():
            # model.load_state_dict(torch.load(saved_model_name, map_location=device))

            for data, label in Test_data:
                pred = model(data.type(torch.FloatTensor).to(device))

                loss = nn.MSELoss()(pred, label.type(torch.FloatTensor).to(device))
                total_loss += loss / len(Test_data)

                pred = pred.cpu().numpy()
                mean = pred.mean()
                predict.append(mean)
            pred_inverse = np.array(predict)
            # pred_inverse = scaler_y.inverse_transform(np.array(predict).reshape(-1, 1))
            # testY_inverse = scaler_y.inverse_transform((testY_tensor).reshape(-1, 1))
            # print(type(predict[0]))
            real_y = np.array(real_y)

        MSE = mean_squared_error(real_y, pred_inverse)
        MAPE = mean_absolute_percentage_error(real_y, pred_inverse)
        R2 = r2_score(real_y, pred_inverse)
        #
        print('MSE SCORE : ', MSE)
        print("MAPE SCORE : ", MAPE)
        print('R2 SOCRE : ', R2)