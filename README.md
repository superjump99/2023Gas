# 2023 가스 예측 

# Update 
- [02/28] 환경설정, 특징, 함수 정리, PCA, Corr 
- [02/03] 전처리 완료한 데이터 셋을 사용하여 Pycaret [Model 성능 업데이트](#사용-모델)
- [01/31] Func_set.py/data_processing()/[const()](#constdfnone)   
  제약조건     
  - CH4 + CO2 + O2 > 100　  
  - Q, P, T outlier　삭제(IQR 사용)  　　
    Q -> (Min : -1.2　Max : 2.53)  
    P -> (Min : -159.0　Max : 353.0)  
    T -> (Min : -10.0　Max : 38.0) 
  - VOR 범위외 (0~100)
    

--------
# Getting Started
- [02/28]  Using pytorch  
```
  conda create -n gas3.9 python=3.9  
  conda activate gas3.9  
  pip install -r requirements3.9.txt  
```   
- using pycaret  
```
  conda create -n gas3.7 python=3.7  
  conda activate gas3.7  
  pip install -r requirements3.7.txt  
  python –m ipykernel install --user --name gas3.7 --display-name "gas3.7"  
```  

## 정리  
  - 과정  
  - 초기 데이터 처리
      - 특정 데이터 타입이 문자열(-,+), 공백(' '), 일부 수치형으로 존재하지 않는(4..7) 데이터 값이 존재 => Nan값 변환  
  - 머신러닝에 사용하는 데이터와 다르게 시계열 데이터로 사용하기 위해서는 
  
  - PCA분석  
    - PCA(주성분 분석)을 통해 차원 축소 실행
    - 입력변수 간 상관관계를 줄이기 위해, 대표하고자 하는 특성을 선별하고 하고 데이터를 이해하기 쉽게하기 위함
    - (1. PCA 결과 2. PCA 결과로 머신에 넣었을 때 결과 필요) : 평균적으로 PCA3수준으로 하여 진행해보았지만 좋은 성능이 나오지 않음  
  - 상관분석 
      - 상관관계 : 어떤 선형적 관계를 갖고 있는 지를 분석하는 방법     
      - 상관 관계는 두 변수간 선형적 관계를 갖고 있는지를 분석, 그러나 -1, 0, 1 값이 유의미한 값을 갖는지 안갖는지 데이터 분포를 보고 직접 판단 할 필요가 있음  
      - (상관 분석 결과 표 형식 필요) : 변수간의   
  - 시계열 사용 모델-> RNN, LSTM, Nlinear 각각 예시에 대한 결과 값  
  - 모델링 과정 : 
  
  - 결과가 안나온 이유  
    1. 차원의 개수    
      - 각 독립변수는 독립적으로 다른 변수에 영향을 끼치지 않도록 해야함, 비슷한 설명과 의미를 가진 특징이 강한 하나의 변수로 사용하는 것이 좋음
    2. 데이터 결측치 처리  
      - 데이터의 평균을 사용하여 진행하였지만, 시계열 데이터 특성상 시간의 흐름으로 데이터를 학습하기에 결측치를 처리하는 과정에서 의도한 목적대로 학습되지 않았을 수 있음
    3. 모델링  
      - 한 장소에 대해 학습을 하고 저장 -> 다음 장소에 대해서 학습한 모델 불러와 다시 학습하는 과정에서 학습된 가중치의 기억을 잃어버림
      - 한 장소에 대해서 제대로 된 모델링을 하는 것이 우선순위가 되야함  
  - 어려웠던 점
    1. 데이터 변환 
      - 데이터를 사용할 수 있게 끔 정리하는 부분에 있어 시간적으로 많은 부분이 걸렸음
    2. 데이터 결측치 처리
      - 결측값과 이상값에 대해서 처리를 할 때 이것 저것 사용해보니까 좋은 결과가 나왔다는 시간적으로 오래걸리며, 이 부분에서 어떻게 의사결정해야 하는지 에 대한 어려움 존재
    3. 데이터 이해
      - 데이터 분포를 보고 각 데이터에 대한 의미를 통해 결측치 처리가 필요로 함
    4. 모델링
      - 모델링 과정에서 2,3번의 부족함을 느끼며 주먹구구식보다는 근거있는 
    
# 특징
[02/28]  

  
- Pycaret 결과는 jupyter notebook 으로 확인 필요
- 사용 가능할 수 있는 Data는 최대한 사용하기 위하여 전처리 과정 수행
  Raw data의 수치형으로 존재하지 않는 오류가 발생  
  \* *위 문제를 해결하기 위한 방법 : [data_processing()](#data_processingfilename)*   
- 결측값이 존재하는 row 대체할 수 있도록 변경  
- Train, Test data 비율 = 전체 데이터의 8 : 2  
- normalization (Min-Max 사용) 전, 후 비교  
- 평가 방법
  MAE,	MSE,	RMSE,	R2,	RMSLE,	MAPE    
  

  

# Model
## 사용 모델
- Pycaret Regression
- DNN  
[02/28]  
- RNN  
- LSTM  
- Nlinear  
\* *각각의 사용 기능은 모델 파일 내부 주석 참조*


------------
# Make Files.py
Raw data를 읽어 사용하여 필요한 File 생성 파일  
1. 2012년 ~ 2016년 시간 순서로 정렬된 5개 파일을 1개의 데이터로 만듬   
2. Raw data의 시간을 DateTime 형식으로 바꿔 날짜 형식 통일
3. 전체 데이터에서 7개의 시트별(`CH4`, `CO2`, `O2`, `P`, `Q`, `T`, `VOR`)로 파일 구분   
\* *사용되는 함수 `Func_set.py` 설명 참조*   

# EvalFunc.py  
모델 성능 평가를 위한 함수 집합 파일  
\* *아래 평가지표 참조*  

# Func_set.py  
Data Process와 Model 내부에서 실행 시키기 위한 함수 집합 파일  
\* *함수 사용 위치와 기능은 아래 설명 참조*  

# Function descriptions in Func_set.py
## In Data Process Folder/Make Files.py
### combine_years_and_split_7files()   
Raw data(2012~2016 포집정데이터)를 시간 순서대로, 데이터 유형별로 정렬하여 7개 파일 생성
- Input : Raw data
- Output : `CH4`, `CO2`, `O2`, `P`, `Q`, `T`, `VOR` 의 7개 csv files
  - 2012년 ~ 2016년 초기 초기 파일을 불러와 **CH4, CO2, O2, P, Q, T, VOR** 총 7개의 file 생성   
  -  사용하기 위한 기초 작업 
  - '변환 데이터' 폴더에 저장    


### cal_date(df, year)
`combine_years_and_split_7files()` 실행 시 , 날짜 형식을 맞춰 주기 위한 함수  
- Input : Raw data의 날짜 형식
- Output : Datatime 형식으로 변환   

### createFolder(directory)
`combine_years_and_split_7files()` 실행 시, 변환 데이터 폴더를 만들기 위한 함수  

------------
## In Model Folder/Pycaret Regression.ipynb, DNN.py   
### time_series_data()
사용 데이터 생성 및 불러오는 함수   
- Input : combine_years_and_split_7files의 7개 파일   
- Output : TSD.xlsx   
  - CH4, CO2, O2, P, Q, T, VOR 총 7개 파일을 1. 장소별로 구분, 2. DateTime(index)유지   
  - '변환데이터/TSD.csv' 존재 시 새로 생성하지 않음     


### data_processing(filename)
기본적인 전처리(사용 가능 데이터 회생)
- Input : 데이터 프레임  
- Output : 회생 및 선택적 전처리 가능하도록 nan값을 사용
- 사용 이유   
  - 데이터 값이 수치로 되어 있지 않은 데이터 존재   
  - 수치로 되어있지만, 특수 문자가 포함되어있는 경우 존재   
  - dtypes is not float   
- 해결 방법  
  - 회생 불가능한 데이터 Nan 처리
  - 회생 가능한 데이터 오류 처리   
  - float 형태 변환      
- 발견된 오류 및 해결 방법   

| 방법 | 발생 오류 | 실제 오류값 | 변환값 |  
|:----|:--------:|:-----------:|:-------:|  
| 삭제 | 관측값이 ` `, `c`, `-`, `+` 인 경우 | c | np.nan |  
| 회생 | 값 처음이 `.`, `+` 으로 시작하는 경우 | .7.3 | 7.3 |  
| 회생 | 값 마지막이  `.`, `+` 인 경우 |  48.7+ | 48.7 |  
| 회생 | 값 안에 `*` 있는 경우 | 2*9.2 | 29.2 |  
| 회생 | 값 안에 `..` 있는 경우 | 8..7 | 8.7|  

### const(df=None)
제약조건 함수
- Gas(CH4, CO2, O2) 합 100 초과 -> CH4, CO2, O2 Nan 처리
- P, T, Q Outlier -> 해당 값 Nan 처리　　

### outlier(df = None,column=None, weight=1.5)
IQR 방법으로 Outlier 찾아 삭제시키는 함수  


#### 평가지표
|평가지표 | 설명 | 특징 |	  
|:------:|:-----:|----|
| MAE <br>(평균 절대 오차) | 오차의 절대값의 합의 평균|장점<br> - 전체 데이터의 학습된 정도를 쉽게 파악 <br> - 이상치에 민감하지 않음 <br><br>단점<br> - 오차 발생 방법과 음수인지 양수인지 알 수 없음 <br> - 함수 값에 미분 불가능한 지점 존재
| MSE <br>(평균 제곱 오차) | 추정된 값과 실제 값 간의 평균 제곱차이 |장점<br> - 실제 정답에 대한 정답률의 오차 뿐만아니라 다른 오차에 대한 정답률의 오차도 포함하여 계산 <br> - 최적값에 가까워질 수록 이동값이 다르게 변화하여 최적값에 수렴하기 용이 <br> - 모든 함수 값에서 미분 가능  <br><br>단점<br> - 제곱하기 때문에 절대값이 1 미만인 값은 더 작아지고, 1보다 큰 값은 더 커지는 **왜곡이 발생**할 수 있음 <br> - 제곱하기에 이상치의 영향을 많이 받음 
| RMSE <br>(평균 제곱근 편차)| MSE에 루트를 씌운 값  |장점<br> - 이상치에 덜 민감, MAE보다는 크고 MSE보다는 작기에 이상치를 잘 다룬다고 간주되는 경향이 있음 <br><br>단점<br> 미분 불가능한 지점을 갖음 |
| RMSLE|  RMSE에 log를 적용해준 지표 | RMSE와 비교해 RMSLE가 가진 장점 <br> -  이상치에 대해 변동이 크지 않음 <br> - 상대적 Error 측정 <br> - 예측값이 실제값보다 작을 때 더 큰 패널티 부여 <br><br>단점<br> - RMSLE의 특성상 실제값이 0이거나 예측값이 0인데 조금이라도 예측값, 실제값과 다르면 큰 페널티|
| MAPE <br>(평균 절대 비율 오차)| MSE의 범위 : 0~무한대 <br> 따라서 MSE의 값이 좋은지 판단하기 어려워 MAPE의 퍼센트 값을 통해 성능평가| 장점<br> - 0~100% 사이의 확률 값을 가지기 때문에 결과 해석이 용이 <br>-  데이터 값의 크기와 관련된 것이 아닌 비율과 관련된 값을 가지기 때문에 모델, 데이터 성능 비교에 용이 <br><br>단점<br> - 실제 정답값에 0이 존재하는 경우 MAPE 계산이 불가능 함 <br> 실제 정답보다 높게 예측했는지 낮게 예측했는지 파악하기 힘듬 <br> 실제 정답이 1보다 작을 경우, 무한대의 값으로 수렴할 수 있음  |    
| R2 <br>(결정계수) | 독립변수가 종속변수에 얼마만큼의 설명력을 갖는지 나타내는 수치 | 다른 지표들은 데이터의 scale에 따라 값이 다르지만 R2은 상대적인 성능이 어느정도인지 직관적으로 판단 가능|   

## 모델 성능
||Model|	MAE	|MSE|	RMSE	|R2|	RMSLE|	MAPE|
|---|---|---|---|---|---|---|---|
rf|	Random Forest Regressor|	3.1155|	18.9958|	4.3583	|0.8953|	0.1802|	0.1482|
et|	Extra Trees Regressor|	3.1410|	19.4178|	4.4065|	0.8930|	0.1836|	0.1486|	
lightgbm|	Light Gradient Boosting Machine|	3.2306|	19.6997|	4.4384|	0.8914|	0.1835|	0.1581|	
knn|	K Neighbors Regressor|	3.2523|	20.5692|	4.5353|	0.8866|	0.1873|	0.1554|	
gbr|	Gradient Boosting Regressor|	3.3639|	21.4960|	4.6364|	0.8815|	0.1992|	0.1740|	
dt|	Decision Tree Regressor|	4.2778|	37.1342|	6.0938|	0.7953|	0.2425|	0.1831|	
lar|	Least Angle Regression|	5.1262|	46.8118|	6.8419|	0.7420|	0.4763|	0.6548|	
br|	Bayesian Ridge|	5.1262|	46.8118|	6.8419|	0.7420|	0.4763|	0.6549|	0.1100|
ridge|	Ridge Regression	|5.1262|	46.8118|	6.8419|	0.7420|	0.4763|	0.6548|	
lr|	Linear Regression	|5.1262|	46.8118|	6.8419|	0.7420|	0.4763|	0.6548|	
lasso|	Lasso Regression	|5.0755|	48.2999|	6.9498|	0.7338|	0.4979|	0.7534|	
huber|	Huber Regressor|	5.0277|	48.4311|	6.9592|	0.7330|	0.5038|	0.7579|	
en| Elastic Net	|5.1955|	53.5689|	7.3190|	0.7047|	0.5250|	0.8517|	
omp|	Orthogonal Matching Pursuit|	5.8579|	61.3293|	7.8313|	0.6619|	0.4910|	0.7777|	
par|	Passive Aggressive Regressor|	6.5489|	75.4391|	8.6235|	0.5840|	0.4895|	0.7148|	
ada|	AdaBoost Regressor	|7.3462|	81.7704|	8.8631|	0.5493	|0.3845|	0.4577|	15.6280|
llar	|Lasso Least Angle Regression	|10.0019|	181.4239|	13.4692|	-0.0000	|0.6769	|1.7056|	
dummy|	Dummy Regressor	|10.0019|	181.4239|	13.4693|	-0.0000|	0.6769|	1.7056|	
