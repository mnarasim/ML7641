#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 20:23:02 2019

@author: Mani
"""

import pandas as pd
from scipy import stats
data = pd.read_csv("heart.csv")

k2, p = stats.normaltest(data.iloc[:,0:-1].values)

alpha = 1e-3

for i in range(len(p)):
    if p[i] < alpha:
        print(data.columns[i], ' -> Null hypothesis can be rejected')
    else:
        print(data.columns[i], ' -> Null hypothesis is accepted')
heart_summary = data.describe()
heart_summary.to_csv('heart_summary.csv')



data = pd.read_csv("Credit_Card.csv")
k2, p = stats.normaltest(data.iloc[:,0:-1].values)

credit_summary = data.describe()
credit_summary.to_csv('creditcard_summary.csv')
for i in range(len(p)):
    if p[i] < alpha:
        print(data.columns[i], ' -> Null hypothesis can be rejected')
    else:
        print(data.columns[i], ' -> Null hypothesis is accepted')