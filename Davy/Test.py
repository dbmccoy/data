from Davy import LinearRegression
from Davy import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# file = [f.split(',') for f in open('ex1data2.txt','r').read().splitlines()]
# table = Table(file)

table = pd.read_csv('train.csv')
# table.columns = ['sqf', 'rooms', 'price']

# table = table.drop(['rooms'], axis=1)

# table.pop(1)
lr = LogisticRegression()
lr.fit(table, 1)
lr.predict()
# errors = [lr.fit(table, i*1) for i in range(500)]

# plt.scatter(lr.features[0],lr.labels)
# print(errors)
# plt.plot(errors)
# plt.show()
