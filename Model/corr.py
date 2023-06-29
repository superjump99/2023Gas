from Func_set import *
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


matrix = np.zeros((7, 7))

# Load the data
# for i in tqdm(range(699)):
df = pd.read_excel('../변환 데이터/TSD.xlsx',sheet_name=0)  #V2B-18
df.set_index('Date Time', inplace=True)
df = data_processing(df)
df = df.fillna(df.mean()['CH4':'VOR'])
# df = df.loc[:,['CH4','CO2','O2']]
corr = df.corr()
corr = np.array(corr)
# matrix += corr

# matrix/=699

print(corr)
# Plot the correlation matrix as a heatmap
plt.imshow(corr, cmap="YlGnBu")
plt.colorbar()

plt.show()
