import joblib                        #load the model and preprocessing objects using Joblib
model=  joblib.load('model.pkl')
x_scaler= joblib.load('x_scaler.pkl')
y_scaler= joblib.load('y_scaler.pkl')

age = int(input("Enter age: "))
sex = int(input("Enter sex (0 = female, 1 = male): "))
bmi = float(input("Enter bmi: "))
children = int(input("Enter number of children: "))
smoker = int(input("Enter whether smoker or not (0 = non-smoker, 1 = smoker): "))
region = int(input("Enter region (0 = southwest, 1 = southeast, 2 = northwest, 3 = northeast): "))

import pandas as pd
new_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],columns=['age','sex','bmi','children','smoker','region'])
print(new_data)
new_data = pd.DataFrame([[18,1,33.770,1,0,2]], columns=['age','sex','bmi','children','smoker','region'])
new_data_scaled = x_scaler.transform(new_data)
insurance_scaled = model.predict(new_data_scaled)
predicted_insurance = y_scaler.inverse_transform(insurance_scaled)  
print('medical insurance cost for a new individual.:',predicted_insurance)