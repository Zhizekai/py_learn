from sklearn import datasets
from numpy import shape
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
print(loaded_data)
data_X = loaded_data.data
data_Y = loaded_data.target

X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2)
model = LinearRegression()
model.fit(X_test, y_test)
Y_pred = model.predict(X_test)
predicted = cross_val_predict(model,data_X,data_Y,cv=10)
plt.scatter(data_Y,predicted,color = 'red' ,marker="*")
plt.scatter(data_Y,data_Y,color = 'black', marker='+')

print(data_Y,predicted)


