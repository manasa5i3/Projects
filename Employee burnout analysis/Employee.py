import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle as pickle
import os
data = pd.read_excel("C:/Users/manas/OneDrive/Documents/MERN/ML projects/employee_burnout_analysis-AI.xlsx")
print(data.head())
print(data.tail())
print(data.describe())
print(data.columns.tolist())
print(data.nunique())
print(data.info())
print(data.isnull().sum())
print(data.isnull().sum().values.sum())
data.corr(numeric_only=True)['Burn Rate'][:-1]
sns.pairplot(data)
plt.show()
data=data.dropna()
data = data.drop('Employee ID', axis = 1)
data.shape
data.dtypes
print(f"Min date{data['Date of Joining'].min()}")
print(f"Max date{data['Date of Joining'].max()}")
data_month=data.copy()

data_month["Date of Joining"] = data_month["Date of Joining"].astype("datetime64[ns]")
data_month["Date of Joining"].groupby(data_month["Date of Joining"].dt.month).count().plot(kind="bar",xlabel="Month", ylabel="Hired Employees")
data_2008=pd.to_datetime(["2008-01-01"]*len(data))
data["Days"]=data["Date of Joining"].astype("datetime64[ns]").sub(data_2008).dt.days
data.Days
#numeric columns
numeric_data=data.select_dtypes(include=np.number)
correlation=numeric_data.corr()['Burn Rate']
print(correlation)
data.corr(numeric_only=True)['Burn Rate'][:-1]
data=data.drop(['Date of Joining','Days'],axis=1)
print(data.head())
cat_columns = data.select_dtypes(include='object').columns
# Use plt.subplots to create a figure and subplots
fig, ax = plt.subplots(nrows=1, ncols=len(cat_columns), sharey=True, figsize=(10, 5))
for i, col in enumerate(cat_columns):
    sns.countplot(x=col, data=data, ax=ax[i])
plt.show()
data=pd.get_dummies(data, columns=['Company Type','WFH Setup Available',
                                   'Gender'],drop_first=True)
data.head()
encoded_columns=data.columns
#split df into x and y
y=data['Burn Rate']
x=data.drop('Burn Rate', axis=1)
#train test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#scale x
scaler=StandardScaler()
scaler.fit(x_train)
x_train=pd.DataFrame(scaler.transform(x_train), index=x_train.index, columns=x_train.columns)
x_test=pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)
import os
import pickle
scaler_filename = 'scaler.pkl'
if not os.path.exists(scaler_filename):
    pickle.dump(scaler, open(scaler_filename, 'wb'))
x_train
y_train
import os
import pickle
path='../data/processed/scaler.pkl'
os.makedirs(os.path.dirname(path), exist_ok=True)
x_train.to_csv(path, index=False)
y_train.to_csv(path, index=False)
linear_regression_model=LinearRegression()
#train
linear_regression_model.fit(x_train,y_train)
print("linear Regression model")
y_pred=linear_regression_model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred,squared=False)
mae=mean_absolute_error(y_test,y_pred)
print("mean squared error: ",mse)
print("root mean squared error: ",rmse)
print("mean absolute error: ",mae)
r2=r2_score(y_test,y_pred)
print("R Squared score: ",r2)