import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection  import train_test_split
import math

a = np.array([1,2,3,4,5,6,7,8,9,10])
test = pd.DataFrame()
test['a'] = a
test['b'] = test['a'].apply(math.sqrt)

X = np.array(test['a']).reshape(-1, 1)

#Initializing Linear Regression object
reg = LinearRegression()
#Test/Train Split
x_train, x_test,y_train,y_test = train_test_split(X, test['b'],test_size =0.2, random_state=1)
reg.fit(x_train, y_train)

print(f"Accuracy: {reg.score(x_test, y_test)}")
