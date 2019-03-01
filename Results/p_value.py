# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:25:11 2019

@author: ie365ta
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:26:43 2017

@author: server
"""

import pandas as pd
import scipy.stats
import numpy as np

data = pd.read_csv('train_loss_old_datasets.csv')
data = data.iloc[:,1:]
number_of_classifers = data.shape[1]
df = pd.DataFrame(np.zeros((number_of_classifers,number_of_classifers)))
df.columns = data.iloc[:,0:data.shape[1]].columns.values
df.index = data.iloc[:,0:data.shape[1]].columns.values
for i in range(0,data.shape[1]):
    for j in range (0,data.shape[1]):
        if i!=j:
            df.iloc[i,j] = scipy.stats.wilcoxon(data.iloc[:,i],data.iloc[:,j],zero_method='wilcox')[1] ###pratt is conservative and wilcox is more standard

df.to_csv("wilcox_pval.csv")