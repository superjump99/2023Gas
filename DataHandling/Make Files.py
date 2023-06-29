from Func_set import *

currentpath = os.getcwd()
print(currentpath)

'''
=================================================
1. -> combine_years_and_split_7files()
초기 파일 각각 필요한 2012~2016년 자료
시간 데이터 타입으로 변환
CH4, CO2, O2, Q, P, T, VOR 7개의 파일로 저장

input > 2012~2016 포집정데이터
output > 7개 csv 파일 (CH4, CO2, O2, Q, P, T, VOR)
=================================================
2. -> time_series_data()
7개의 파일로 저장한 파일 시계열 데이터로 변환
장소별로 CH4, CO2, O2, Q, P, T, VOR 값을 저장

input > 7개 파일 (CH4, CO2, O2, Q, P, T, VOR)
output > 1개 xlsx파일 (TSD)
=================================================
'''

combine_years_and_split_7files()
time_series_data()