import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1","G2","G3","studytime","failures","absences"]]

predict = "G3" #final grade

x = np.array(data.drop([predict],1)) # training data : G3 is dropped
y = np.array(data[predict]) 
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size= 0.1)
'''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size= 0.1)

    # train
    linear = linear_model.LinearRegression()
        
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
        best = accuracy
        # save to pickle
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
'''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


print("Coefficient: ", linear.coef_) #5Ds = 5 coefficient
print("Intercept: ", linear.intercept_) # y

predicitons = linear.predict(x_test)

for x in range(len(predicitons)):
    print(predicitons[x], x_test[x], y_test[x])

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
