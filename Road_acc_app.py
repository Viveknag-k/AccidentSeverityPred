import streamlit as st
import numpy as np
import pandas as pd
import pickle
model = pickle.load(open("rf.pkl","rb"))
st.header("Road Accidents Severity Classification")
st.write("Give the input features : ")
def ip_features():
    Day_of_week = st.selectbox("Day_of_the_week",['Monday' ,'Sunday', 'Friday', 'Saturday' ,'Thursday', 'Tuesday' ,'Wednesday'])
    Road_surface_type = st.selectbox("Road_surface_type",['Asphalt roads', 'Earth roads', 'Asphalt roads with some distress',
 'Gravel roads' ,'Other'])
    Type_of_collision = st.selectbox("Type_of_collisison",['Collision with roadside-parked vehicles',
 'Vehicle with vehicle collision', 'Collision with roadside objects',
 'Collision with animals' ,'Other', 'Rollover', 'Fall from vehicles',
 'Collision with pedestrians', 'With Train' ,'Unknown'])
    Number_of_vehicles_involved = st.number_input("Number_of_vehicles_involved")
    Number_of_casualties=st.number_input("Number_of_casualties")
    Casualty_class=st.number_input("Casualty_class")
    Sex_of_casualty=st.selectbox("Sex",['Male','Female'])
    Age_band_of_casualty = st.selectbox("age band",['31-50' ,'18-30' ,'Under 18', 'Over 51', '5'])
    Casualty_severity = st.selectbox("Casualty_severity",[1,2,3])
    
    data = {'Day_of_week':Day_of_week,
            'Road_surface_type':Road_surface_type,
            'Type_of_collision':Type_of_collision,
            'Number_of_vehicles_involved':Number_of_vehicles_involved,
            'Number_of_casualties':Number_of_casualties,
            'Casualty_class':Casualty_class,
            'Sex_of_casualty':Sex_of_casualty,
            'Age_band_of_casualty':Age_band_of_casualty,
            'Casualty_severity':Casualty_severity}
    features= pd.DataFrame(data,index=[0])
    return features
df = ip_features()

def main():
    from sklearn.preprocessing import LabelEncoder
    l = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = l.fit_transform(df[col])
    if st.button('Predict'):
        pred = model.predict(df)
        #if st.success("The person got {}:".format(pred))
        if pred == 1:
            st.success("The person got Slight Injury")
        elif pred == 2:
            st.success("The person got Serious Injury")
        elif pred == 3:
            st.success("The person got Fatal Injury")
                
if __name__ == '__main__':
    main()

   