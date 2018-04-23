# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('timeseries_small.csv')
data.drop(['date'],axis=1)
index = data.index.values

#Visualize the sales data
plt.figure()
plt.plot(index, data['sales'])
plt.title('Real Sales')
plt.show()

#Computet he moving average and save it to new column in data matrix
data['movavg'] = data['sales'].rolling(window=3,center=True,min_periods=1).mean()

plt.figure()
plt.plot(index, data['movavg'])
plt.title('Average Sales')
plt.show()

#show both plots in overlapping manner to check if the average compliments the real sales or not
plt.figure()
plt.plot(index, data['sales'])
plt.plot(index, data['movavg'])
plt.title('Avg. Sales v/s Real Sales')
plt.show()

# Now, split the data in two parts
# First 29 days data for training
# and 30th day data for testing
X_train = np.zeros((29,1))
X_test = np.zeros((1,1))
X_train[:,0] = data.index.values[0:29]
y_train = data.sales.values[0:29]
X_test[:,0] = data.index.values[29:]
y_test = data.sales.values[29:]
y_test_avg = data.movavg.values[29:]

#Now train & test the linear regression model with sales data
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
y_preds = reg.predict(X_test)

#Show the estimated values
print('Estimated sales volume for 30th day is : ' + str(y_preds))
print('Original sales volume for 30th day is : ' + str(y_test))
print('Average sales volume for 30th day is : ' + str(y_test_avg))
