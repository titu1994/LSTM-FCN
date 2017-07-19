import pandas as pd
import scipy.stats
import numpy as np

data = pd.read_csv("utils/summary_of_dataset.csv")

number_of_classifers = data.shape[1]-6
df = pd.DataFrame(np.zeros((number_of_classifers,number_of_classifers)))
df.columns = data.iloc[:,6:data.shape[1]].columns.values
df.index = data.iloc[:,6:data.shape[1]].columns.values
for i in range(6,data.shape[1]):
    for j in range (6,data.shape[1]):
        df.iloc[i-6,j-6] = scipy.stats.wilcoxon(data.iloc[:,i],data.iloc[:,j],zero_method='pratt')[1] ###pratt is conservative and wilcox is more standard

df.to_csv("utils/summary_of_pvalue.csv")