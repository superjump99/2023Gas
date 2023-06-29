import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from Func_set import *

'''
참고 사이트 
PCA, 주성분의 개수는 어떤 기준으로 설정할까?
https://techblog-history-younghunjo1.tistory.com/134
'''
#
# df = pd.read_excel('./변환 데이터/TSD.xlsx', sheet_name=0)
#
# df.set_index('Date Time', inplace=True)
#
# df = data_processing(df)
# df = df.fillna(df.mean()['CH4':'VOR'])
# df = df.drop(['CH4'],axis=1)
# print(df)
# # 변수 단위 표준화
# minmax_df = MinMaxScaler().fit_transform(df)
# minmax_df = pd.DataFrame(minmax_df,index=df.index, columns=df.columns)
# df = minmax_df
# print(df.head())
#
# # PCA 수행
# pca = PCA(n_components=6) #n_components=7
# pca_array = pca.fit_transform(df)
# print(pca_array)
# pca_df = pd.DataFrame(pca_array, index=df.index,
#                       columns=[f'pca{num+1}' for num in range(6)])
# print(pca_df.head())
# # plt.plot(pca_df)
# # plt.show()
#
# # 기여율 계산
# result = pd.DataFrame({'설명가능한 분산 비율(고윳값)':pca.explained_variance_,
#                        '기여율':pca.explained_variance_ratio_},
#                       index=np.array([f'pca{num+1}' for num in range(df.shape[1])]))
# result['누적기여율'] = result['기여율'].cumsum()
# print(result)
#
# plt.plot(result.index,result.iloc[:,0])
# plt.xlabel("Number of PCA")
# plt.ylabel("Cumulative Explained Variance")
# plt.show()


place = 699 # 전체 매립 가스장
for i in tqdm(range(place)):
    count = 0
    df = pd.read_excel('./변환 데이터/TSD.xlsx', sheet_name=i)

    df.set_index('Date Time', inplace=True)

    df = data_processing(df)        # 데이터
    df = df.fillna(df.mean()['CH4':'VOR'])
    df = df.drop(['CH4'],axis=1)

    # 변수 단위 표준화
    minmax_df = MinMaxScaler().fit_transform(df)
    minmax_df = pd.DataFrame(minmax_df, index=df.index, columns=df.columns)
    df = minmax_df
    # print(df.head())

    # PCA 수행
    pca = PCA()  # n_components=7
    pca_array = pca.fit_transform(df)

    pca_df = pd.DataFrame(pca_array, index=df.index,
                          columns=[f'pca{num + 1}' for num in range(df.shape[1])])
    # print(pca_df.head())

    # 기여율 계산
    result = pd.DataFrame({'설명가능한 분산 비율(고윳값)': pca.explained_variance_,
                           '기여율': pca.explained_variance_ratio_},
                          index=np.array([f'pca{num + 1}' for num in range(df.shape[1])]))
    result['누적기여율'] = result['기여율'].cumsum()

    if count == 0 :
        accmulate = result
        count +=1
    else:
        accmulate = accmulate.add(result)
accmulate = accmulate/len(place)
print(accmulate)
plt.plot(accmulate.index,accmulate.iloc[:,0])
plt.xlabel("Number of PCA")
plt.ylabel("Cumulative Explained Variance")
plt.show()