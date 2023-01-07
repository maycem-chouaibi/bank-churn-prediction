# Created By: Mayssem Chouaibi
# Description: churn prediction for banking customers
# Dataset available on Kaggle

import dataHelper

data = dataHelper.load('churn.csv')

data2 = dataHelper.prepareData(data)

x = data2.drop('Exited', axis='columns')
y = data2['Exited']
