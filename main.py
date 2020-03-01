import numpy as np
import pandas as pd
from models import NN

df = pd.read_csv('data1.csv')
X = df[['X1', 'X2']]
X = np.array(X).transpose()

Y = df['y']
Y = np.array(Y).reshape(100,1).transpose()

model = NN(4, 0)
params, cost, iterations = model.fit(X, Y, alpha=0.001, epochs=100000, plot_cost=True, verbose=True)
res = model.predict(X, params)
print("Training accuracy is {} %".format(model.accuracy(res, Y)))
