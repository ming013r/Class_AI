import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

train =np.loadtxt('click.csv', delimiter = ',', dtype = 'int', skiprows = 1)
df = pd.dataframe(train)
print(df)
train_x = df[:,0]
train_y = df[:,1]
plt.scatter(train_x, train_y)
plt.show()



theta0 = np.random.rand()
theta1 = np.random.rand()

def f(x):
    return theta0 + theta1 * x

def E(x, y):
    return 0.5 * np.sum((y-f(x))**2)

ETA = 1e-3
diff = 1
count = 0

error = E(train_x, train_y)
while diff > 1e-2:
    tmp_theta0 = theta0 - ETA * np.sum((f(train_x)-train_y))
    tmp_theta1 = theta1 - ETA * np.sum((f(train_x)-trainy)*train_x)
    
    theta0 = tmp_theta0
    theta1 = tmp_theta1
    
    current_error = E(train_x, train_y)
    diff = error = current_error
    
    log = '{}次: theta0 = {:(3f),theta1 = {:,3f},差分=(:4f)}'
    print(log.format(count, theta0, theta1,diff))