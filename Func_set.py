import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm
# from numba import jit


years= [2012, 2013, 2014, 2015, 2016]
columns = ['CH4', 'CO2', 'O2', 'Q', 'P', 'T', 'VOR']

# using make file
def combine_years_and_split_7files():
    main_df = pd.DataFrame()
    for sheet_name in columns:
        print(f"----{sheet_name} file 생성----")
        for year in years:
            print(f"Read {year}포집정별.xlsm")
            read_excel = pd.read_excel(f"../포집정데이터 취합/{year}포집정별.xlsm", sheet_name=sheet_name,
                                       engine="openpyxl")  # sheets의  7번 진행해야함

            month = read_excel[year].ffill()  # Month의 NaN 값을 표시하기 위함

            df = read_excel.drop([year], axis=1)  # 기존 데이터 프레임의 월을 의미하는 NaN 값 삭제
            df = pd.concat([month, df], axis=1)  # Month 시리즈와 df 병합
            df.rename(columns={'Unnamed: 1': 'day'}, inplace=True)  # 일자의 column을 day로 변경

            ###################################### 날짜 계산
            df = cal_date(df, year)
            if year == years[0]:
                main_df = df
            else:
                main_df = pd.concat([main_df, df])
        createFolder('../변환 데이터')
        main_df.to_csv(f'../변환 데이터/{sheet_name}.csv', index=False)
        print(f"{sheet_name} file 생성 완료")
def cal_date(df, year):
    date_list = []
    for n in range(len(df)):
        try:
            month = str(int(df[year].loc[n]))
        except:
            if '월' in df[year].loc[n]:
                month = df[year].loc[n][:-1]
            else:
                month = df[year].loc[n]
        day = str(df["day"].loc[n])

        if len(month) != 2:
            month = "0" + month
        if len(day) != 2:
            day = "0" + day
        date = str(year) + "-" + month + "-" + day
        date_time = datetime.datetime.strptime(date, '%Y-%m-%d')
        date_list.append(date_time)
    df.insert(0, "date", date_list)
    df.drop([year, 'day'], axis=1, inplace=True)
    return df

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

# @jit
def data_processing(df):
    columns = list(df.columns)
    for col in columns:
        for i in range(len(df[col])):  # len(df[col]
            val = df[col].iloc[i]
            if val == 0.0:
                if col == 'O2' or col == 'VOR':
                    continue
                df.iloc[i] = np.nan
                continue
            if type(val) is int:
                df[col].iloc[i] = float(df[col].iloc[i])  # float 타입으로 변환
            elif type(val) is str:
                if val == '공사중' or val == ' ' or val == 'c' or val == '-' or val == "+":
                    df[col].iloc[i] = np.nan
                    continue
                if "." or "+" == val[0]:
                    df[col].iloc[i] = df[col].iloc[i].lstrip(".")
                    df[col].iloc[i] = df[col].iloc[i].lstrip("+")

                if '.' or "+" == val[-1]:
                    df[col].iloc[i] = df[col].iloc[i].rstrip(".")
                    df[col].iloc[i] = df[col].iloc[i].rstrip("+")

                if "/" in val:
                    change = val.replace("/", '')  # 오류 값을 숫자로 읽을 수 있게 변형
                    df[col].iloc[i] = df[col].iloc[i].replace(df[col].iloc[i], change)  # 특정 위치 값 변환 값 적용

                if "*" in val:
                    change = val.replace("*", '')  # 오류 값을 숫자로 읽을 수 있게 변형
                    df[col].iloc[i] = df[col].iloc[i].replace(df[col].iloc[i], change)  # 특정 위치 값 변환 값 적용

                if ".." in val:
                    change = val.replace("..", '.')
                    df[col].iloc[i] = df[col].iloc[i].replace(df[col].iloc[i], change)  # 특정 위치 값 변환 값 적용

                df[col].iloc[i] = float(df[col].iloc[i])  # float 타입으로 변환
        df[col] = df[col].astype(dtype='float64')
    const(df)
    return df


def const(df=None):
    # 제약조건1 = gas 합 100 이하 , VOR 0~100
    gascount, VORcount = 0,0
    gasmax = 100.000001
    for i in range(len(df)):  # len(df)
        idx = df.iloc[i]

        # gas 합 100 이하
        gassum = idx["CH4"] + idx["CO2"] + idx["O2"]
        if gassum > gasmax:
            gascount +=1
            # print(i, gassum)
            df["CH4"].iloc[i] = np.nan
            df["CO2"].iloc[i] = np.nan
            df["O2"].iloc[i] = np.nan
        # VOR 0 ~ 100
        if idx["VOR"] > 100:
            VORcount +=1
            df["VOR"].iloc[i] = np.nan
    # print(f"gas 합 100 초과 개수 : {gascount}")
    # print(f"VOR 범위 외 개수 : {VORcount}")

    # 제약조건2 = outlier 찾기
    outlier(df = df,column="P")         # P outlier
    outlier(df = df,column="T")         # T outlier
    outlier(df = df,column="Q")         # Q outlier


# 제약조건 2. 아웃라이어 min-max 찾기
def outlier(df = None,column=None, weight=1.5):
    level_1Q = df[column].quantile(0.25)
    level_3Q = df[column].quantile(0.75)
    IQR = level_3Q-level_1Q
    IQR_weight = IQR * weight
    lowest = level_1Q - IQR_weight
    highest = level_3Q + IQR_weight

    count = 0
    for i in range(len(df)): #len(df)
        idx = df.iloc[i]
        if idx[column] < lowest or idx[column] > highest:
            count += 1
            df[column].iloc[i] = np.nan

def time_series_data():
    file = '../변환 데이터/TSD.xlsx'
    if os.path.isfile(file):
        print('exist `TSD.xlsx` file')
    else:
        excel_writer = pd.ExcelWriter('../변환 데이터/TSD.xlsx', engine='xlsxwriter')
        init_df = pd.read_csv(f'../변환 데이터/CH4.csv')
        count_Place = len(init_df.keys())
        for i in tqdm(range(count_Place-1)):
            sheet_df = pd.DataFrame()
            for column in columns:
                read_file = f'../변환 데이터/{column}.csv'
                df = pd.read_csv(read_file)
                datetime = df.iloc[:, 0]
                if column == columns[0]:
                    sheet_df["Date Time"] = datetime

                space = df.keys()[i+1]
                sheet_df[f"{column}"] = df.loc[:,space]
            sheet_df.to_excel(excel_writer, index=False, sheet_name=space)
        excel_writer.save()