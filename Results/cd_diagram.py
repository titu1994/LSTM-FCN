# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 19:37:36 2017

@author: server
"""


import Orange
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


data = pd.read_csv('train_loss_old_datasets.csv')
df = data.iloc[:,1:]

number_of_classifers = df.shape[1]

meanvalues = np.array(df.rank(1,method = 'min',ascending=False).mean(0))
geomean_lstm = stats.gmean(df.rank(1,method = 'min',ascending=False).iloc[:,1])
geomean_alstm = stats.gmean(df.rank(1,method = 'min',ascending=False).iloc[:,0])
names = df.columns.values
cd = Orange.evaluation.compute_CD(meanvalues, df.shape[0]) #tested on 30 datasets
Orange.evaluation.graph_ranks(meanvalues, names, cd=cd, width=7, textspace=1.25)
plt.show()
