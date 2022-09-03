import pandas as pd
import numpy as np
import streamlit as st
print("Running")
df=pd.read_csv('RAW_DATA.csv')
df.drop(columns=['ID','RACE','POSTAL_CODE','DUIS','CHILDREN'],axis=1,inplace=True)
print(df.info())

print('Number of columns {} Number of rows {}'.format(df.shape[1],df.shape[0]))
# print('Total NUll values',df.isna().sum())
df['CREDIT_SCORE']=df['CREDIT_SCORE'].fillna(df['CREDIT_SCORE'].median())
df['ANNUAL_MILEAGE'].fillna(df['ANNUAL_MILEAGE'].median(),inplace =True)
print(df['ANNUAL_MILEAGE'].describe())
print(df.info())
# cleaning gender
df['GENDER']=df['GENDER'].replace("male","1")
df['GENDER']=df['GENDER'].replace("female","0")
df['GENDER']=df['GENDER'].astype(int)
# gender cleaned

# ceaning df['DRIVING_EXPERIENCE']
df['DRIVING_EXPERIENCE']= df['DRIVING_EXPERIENCE'].str.replace("y","")
# df['DRIVING_EXPERIENCE'].replace("y","")
print(df['DRIVING_EXPERIENCE'].unique())




# cleaning eduaction None=0 high school=1 and university =2
df['EDUCATION']=df['EDUCATION'].str.replace("none","0")
df['EDUCATION']= df['EDUCATION'].str.replace("high school","1")
df['EDUCATION']= df['EDUCATION'].str.replace("university","2")
df['EDUCATION']= df['EDUCATION'].astype(int)
print(df['EDUCATION'].unique())
# Done!

#cleaning Income  upperclass=3 , poverty=0,working class=1,middle class=2
df['INCOME']=df['INCOME'].str.replace("upper class","3")
df['INCOME']=df['INCOME'].str.replace("middle class","2")
df['INCOME']=df['INCOME'].str.replace("working class","1")
df['INCOME']=df['INCOME'].str.replace("poverty","0")
df['INCOME']=df['INCOME'].astype(int)
print(df['INCOME'].unique())
# income cleaned

# VEHICLE_OWNERSHIP will change to 1-yes mine and 0-not mine
df['VEHICLE_OWNERSHIP']=df['VEHICLE_OWNERSHIP'].astype(int)
print(df['VEHICLE_OWNERSHIP'].unique())
# VEHICLE_OWNERSHIP cleaned

#'VEHICLE_YEAR' before 2015 =0 and after 2015 =1 
df['VEHICLE_YEAR']=df['VEHICLE_YEAR'].str.replace("before 2015","0")
df['VEHICLE_YEAR']=df['VEHICLE_YEAR'].str.replace("after 2015","1")
df['VEHICLE_YEAR']=df['VEHICLE_YEAR'].astype(int)
print(df['VEHICLE_YEAR'].unique())
# cleaned VEHICLE_YEAR

#  MARRIED cleaning
df['MARRIED']=df['MARRIED'].astype(int)
print(df['MARRIED'].value_counts())
#  MARRIED cleaned

# mileage cleaninng
df['ANNUAL_MILEAGE']=df['ANNUAL_MILEAGE'].astype(int)
print(df['ANNUAL_MILEAGE'].unique())
# mileage cleaned

# VEHICLE_TYPE  sedan=0  sports car=1
df['VEHICLE_TYPE']=df['VEHICLE_TYPE'].str.replace("sedan","0")
df['VEHICLE_TYPE']=df['VEHICLE_TYPE'].str.replace("sports car","1")
df['VEHICLE_TYPE']=df['VEHICLE_TYPE'].astype(int)
print(df['VEHICLE_TYPE'].unique())
# vehicle type cleaned

# outcome cleaning
df['OUTCOME']=df['OUTCOME'].astype(int)
# outcome cleaned


# cleaning DRIVING_EXPERIENCE "0-9" =0,"10-19"=1,"20-29"=2,"30+"=3
df['DRIVING_EXPERIENCE']=df['DRIVING_EXPERIENCE'].replace("0-9","0")
df['DRIVING_EXPERIENCE']=df['DRIVING_EXPERIENCE'].replace("10-19","1")
df['DRIVING_EXPERIENCE']=df['DRIVING_EXPERIENCE'].replace("20-29","2")
df['DRIVING_EXPERIENCE']=df['DRIVING_EXPERIENCE'].replace("30+","3")
df['DRIVING_EXPERIENCE']=df['DRIVING_EXPERIENCE'].astype(int)
print(df['DRIVING_EXPERIENCE'].unique())
# DRIVING_EXPERIENCE cleaned

# AGE cleaning '16-25'=0,'26-39'=1,'40-64'=2,'65+'=3
df['AGE']=df['AGE'].str.replace("+","") 
df['AGE']=df['AGE'].str.replace("16-25","0")
df['AGE']=df['AGE'].str.replace("26-39","1")
df['AGE']=df['AGE'].str.replace("40-64","2")
df['AGE']=df['AGE'].str.replace("65","3")
df['AGE']=df['AGE'].astype(int)
print(df['AGE'].unique())
# AGE cleaned

print(df.info())
df.to_csv('clean_data.csv')
