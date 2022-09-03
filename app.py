import pandas as pd
import numpy as np
import streamlit as st
# ______________ALL DATA CLEANED for MODEL________________________

data=pd.read_csv('clean_data.csv')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data.drop(columns=['Unnamed: 0'],axis=1,inplace=True)


x=data[['AGE', 'GENDER', 'DRIVING_EXPERIENCE', 'EDUCATION', 'INCOME',
       'CREDIT_SCORE', 'VEHICLE_OWNERSHIP', 'VEHICLE_YEAR', 'MARRIED',
       'ANNUAL_MILEAGE', 'VEHICLE_TYPE', 'SPEEDING_VIOLATIONS',
       'PAST_ACCIDENTS']]
y=data.OUTCOME

X_train,x_test,Y_train,y_test=train_test_split(x,y,test_size=0.2)



#RandomForestClassifier
rn_model=RandomForestClassifier()
rn_model.fit(X_train,Y_train)
pr_rn_reg=rn_model.predict(x_test)
# print("ACCURACY OF RANDOM FOREST : {}%".format(np.round(100*accuracy_score(y_test,pr_rn_reg))))
 

# _____________________MODELLING END___________________________

# STREAMLIT PART MAKING UI:
st.warning(" ")
st.title("Car Insurance Claim")
col1,col2=st.columns(2)
with col1:
       st.write('This Predictive model is all about to check how probabale is that a person  will claim their car insurnace. ')
       st.header('How This helps?')
       st.info('')
       st.write('For a **User**, it will mean that he or she is more prone to accidents and should drive safely by keeping in mind that the probability of meeting an accident or harming someone is higher.')
       st.info('')
       st.write('For the **Insurers**, this model can help them to see how much their cost is risked against a certain amount of premium for which a customer is insured so that they can adjust their services accordingly.')
with col2:
       st.image('bg.jpg',width=500)

st.warning(" ")

st.write("### Let's Predict.....")

col1,col2,col3=st.columns(3)
with col1:
       g=st.selectbox("Gender",["","Male","Female"])
       if(g=="Male"):
              g=1
       if(g=="Female"):
              g=0
       
       a=st.selectbox("Select Your Age",['','16-25 yrs','26-39 yrs','40-64 yrs','65+ yrs'])
       if(a=="16-25 yrs"):
              a=0
       if(a=="26-39 yrs"):
              a=1
       if(a=="40-64 yrs"):
              a=2
       if(a=="65+ yrs"):
              a=3
       
       e=st.selectbox("Education",["","High School","University","None"])
       if(e=="High School"):
              e=1
       if(e=="University"):
              e=2
       if(e=="None"):
              e=3
       
       m=st.selectbox("Married",["","Yes","No"])
       if(m=="Yes"):
              m=1
       if(m=="No"):
              m=0

       
with col2:
       v_o=st.selectbox("Vehicle Ownwership",["","Owner","Not the Owner"])
       if(v_o=="Owner"):
              v_o=1
       if(v_o=="Not the Owner"):
              v_o=0
       
       v_y=st.selectbox("Vehicle Year",["","Before 2015","After 2015"])
       if(v_y=="Before 2015"):
              v_y=0
       if(v_y=="After 2015"):
              v_y=1

       v_t=st.selectbox("Vehicle Type",["","Sedan","Sports"])
       if(v_t=="Sedan"):
              v_t=0
       if(v_t=="Sports"):
              v_t=1

       d_e=st.selectbox("Driving Experience",["","0-9 yrs","10-19 yrs","20-29 yrs","30+ yrs"])
       if(d_e=="0-9 yrs"):
              d_e=0
       if(d_e=="10-19 yrs"):
              d_e=1
       if(d_e=="20-29 yrs"):
              d_e=2
       if(d_e=="30+ yrs"):
              d_e=3

with col3:
       i=st.selectbox("Income Class",["","Upper","Middle","Working","Poverty"])
       if(i=="Upper"):
              i=3
       if(i=="Middle"):
              i=2
       if(i=="Working"):
              i=1
       if(i=="Poverty"):
              i=0

       credit_score=st.number_input("Enter Credit score")
      
       annual_mileage=st.number_input("Enter Annual Mileage of Your Car(kms)")

       s_v=st.selectbox('*Ever confronted for Traffic Violations',["","Yes","No"])
       if(s_v=="Yes"):
              s_v=1
       if(s_v=="No"):
              s_v=0

acc_dnts=st.selectbox("Any *PAST ACCIDENTS* with your vehicle",["","Yes,I have encountered","No,I haven't encountered"])
if(acc_dnts=="Yes,I have encountered"):
       acc_dnts=1
if(acc_dnts=="No,I haven't encountered"):
       acc_dnts=0

inputt=[a , g , d_e , e , i , credit_score , v_o , v_y , m , annual_mileage , v_t , s_v , acc_dnts]
print(inputt)
bn=st.button("Predict")
if bn:
       if rn_model.predict([inputt])==1:
              st.success("#### 85%  chances  of * Claiming * the Insurance")
       if rn_model.predict([inputt])==0:
              st.error("#### 85%  chances  of * Not Claiming *the Insurance")
st.warning("")
