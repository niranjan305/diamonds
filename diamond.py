# Loading standard libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from sklearn.preprocessing import scale
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Import data from diamonds csv file
data = pd.read_csv('diamonds.csv')

data = data.reindex(columns=['carat','color','cut','clarity','depth','table','price','x','y','z'])

#Convert categorical properties to numbers
data.cut = pd.Categorical(data.cut)
data['cut']=data['cut'].cat.codes

data.color = pd.Categorical(data.color)
data['color']=data['color'].cat.codes

data.clarity = pd.Categorical(data.clarity)
data['clarity']=data['clarity'].cat.codes

# Assigning X and y
X = pd.DataFrame(data,columns=['carat','color','cut','clarity','depth','table','x','y','z'])
y = pd.DataFrame(data,columns=['price'])

# Dividing data for testing and validation 
X_test = X[:30000]
X_val = X[30000:]

y_test = y[:30000]
y_val = y[30000:]

# Plotting all the quantitative predictors against each other
sns.pairplot(data[['carat','color','cut','clarity','depth','table','price']]);