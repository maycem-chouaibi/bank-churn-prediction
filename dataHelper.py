import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load(fileName):
    # Read data
    return(pd.read_csv(fileName))

def prepareData(data):
    #Preprocessing
    data.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

    scaler = MinMaxScaler()
    columnsToScale = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

    data[columnsToScale] = scaler.fit_transform(data[columnsToScale])
    data['Gender'].replace({'Female': 1, 'Male':0}, inplace=True)

    return(data.drop(columns=['Geography']))

