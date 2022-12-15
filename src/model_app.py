import streamlit as st
import joblib

import request

model = joblib.load("model.joblib")

st.title("House price prediction")
size = st.number_input("Size (in m2)", min_value=0, max_value=1000, value=100)
nb_rooms = st.number_input("Number of rooms", min_value=0, max_value=10, value=2)
garden = st.checkbox("Garden")
orientation = st.selectbox("Orientation", ["North", "South", "East", "West"])

if st.button("Predict"):
    try:
        y_pred = request.predict_request(size, nb_rooms, garden, orientation)
        st.write(f"Predicted price: {y_pred['y_pred']:.0f} €")
    except:
        st.write("Error: could not connect to the API")

nb_samples = st.number_input("Number of samples", min_value=0, max_value=10000, value=1000)
if st.button("Retrain model"):
    try:
        request.retrain_request(nb_samples)
        st.write("Model retrained")
    except:
        st.write("Error: could not connect to the API")

col1, col2, col3 = st.columns([2,6,1])

with col1:
    st.write("")

with col2:
    st.image("../data/logo.jpg", width=400)

with col3:
    st.write("")