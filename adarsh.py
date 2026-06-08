 
import pandas as pd
data=pd.read_csv("insurance.csv")
print(data.head())

from sklearn.preprocessing import LabelEncoder             #labelencoder
label_encoder=LabelEncoder()
data['sex']=label_encoder.fit_transform(data['sex'])
data['smoker']=label_encoder.fit_transform(data['smoker'])
data['region']=label_encoder.fit_transform(data['region'])
x=data.drop(columns=["charges"])
y=(data[['charges']])
print(x)

from sklearn.preprocessing import StandardScaler           #standardscaler
x_scaler=StandardScaler()
y_scaler=StandardScaler()
x_scaled=x_scaler.fit_transform(x)
y_scaled=y_scaler.fit_transform(y)
#print(f"X Scaled:\n{x_scaled}")

from sklearn.model_selection import train_test_split      #training the data
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y_scaled,test_size=0.2,random_state=42)
#print(x_train.shape)

from sklearn.linear_model import LinearRegression         #linear regression model
model=LinearRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
print('coefficient :',model.coef_)
print('intercept:',model.intercept_)

from sklearn.metrics import mean_squared_error             #regression metrics
error=mean_squared_error(y_test,y_predict)
print('mean square error :',error)

import numpy as np
rmse=np.sqrt(error)
print(f"Root Mean Squared Error: {rmse}")

import joblib                                             #save the model and preprocessing objects using Joblib
joblib.dump(model, "model.pkl") 
joblib.dump(x_scaler, 'x_scaler.pkl')
joblib.dump(y_scaler, 'y_scaler.pkl')