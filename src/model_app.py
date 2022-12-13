import streamlit as st
import joblib

import request

model = joblib.load("model.joblib")

st.title("House price prediction")
size = st.number_input("Size (in m2)", min_value=0, max_value=1000, value=100)
nb_rooms = st.number_input("Number of rooms", min_value=0, max_value=10, value=2)
garden = st.checkbox("Garden")

if st.button("Predict"):
    try:
        y_pred = request.predict_request(size, nb_rooms, garden)
        st.write(f"Predicted price: {y_pred['y_pred']:.0f} €")
    except:
        st.write("Error: could not connect to the API")