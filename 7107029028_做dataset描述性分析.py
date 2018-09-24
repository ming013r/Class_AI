
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

df= datasets.load_diabetes()

X = pd.DataFrame(df.data, columns=df.feature_names)

age_ls = np.array(X['age'])
age_mean = age_ls.mean()
age_median = np.median(age_ls)


bmi_ls = np.array(X['bmi'])
bmi_mean = bmi_ls.mean()
bmi_median = np.median(bmi_ls)


plt.scatter(age_ls, bmi_ls)
plt.show()