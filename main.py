import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import pickle

a = pd.read_csv('salary.csv')

a.test = a.test.fillna(a.test.median())

print(a)

reg = linear_model.LinearRegression()
reg.fit(a[['test', 'interview']], a.salary)


with open('model_pickle', 'wb') as f:
 pickle.dump(reg, f)

with open('model_pickle', 'rb') as f:
 k = pickle.load(f)

 print(k.predict([[9, 10]]))
