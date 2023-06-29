from Func_set import *
from pycaret.regression import *
import os

currentpath = os.getcwd()
print(currentpath)

# data read
NTSD = non_time_series_data()
data = data_processing(NTSD)
# print(data.head())
print(data.isnull().sum())



# 데이터 전처리
data = data.dropna()


setting = setup(data=data,
          target='CH4',
          normalize=False,  # False
#         normalize_method = 'minmax', # 기본은 zscore
          transformation=True,  # 데이터 샘플들의 분포가 정규분포에 더 가까워지도록 처리
          fold=5,  # 기본적으로 10 fold 로 training 한다.
          fold_shuffle=True,
#         ignore_features = ['PriceDiff'], # 제외할 컬럼 (이거 너무 편하다!)
#         numeric_features = ['PriceCH','PriceMM'],
#         categorical_features = ['Store7'], # 지정하면 onehotencoding된다.
#         date_features = [], # 날짜 feature를 년월일시 로 바꿔서 onehotencoding 해준다.
          silent=True,  # setup 시 중간에 피쳐속성 확인하고 엔터 쳐줘야하는데 알아서 넘어가게 해준다.
          session_id=123,  # random state number 지정
          use_gpu=True,  # gpu 사용 옵션
#         feature_selection = True,
#         feature_selection_method = 'classic', # or 'boruta'
# classic 은 permutation importance 기반이다.
#         fix_imbalance = True, # data imbalance 를 sampling method로 보정
#         fix_imbalance_method = imblearn.OverSampling.RandomOverSampler()
# 기본은 SMOTE
#         custom_pipeline = pipe, preprocess =False
# 두 개는 세트, 사용자가 원하는 파이프라인을 구성할 수 있다.
          )
print(setting)
model = models()
best = compare_models()

print(best)