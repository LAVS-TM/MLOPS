import streamlit as st
import joblib

model = joblib.load("model.joblib")

st.title("House price prediction")
size = st.number_input("Size (in m2)", min_value=0, max_value=1000, value=100)
nb_rooms = st.number_input("Number of rooms", min_value=0, max_value=10, value=2)
garden = st.checkbox("Garden")
X = [[size, nb_rooms, garden]]

if st.button("Predict"):
    y_pred = model.predict(X)
    st.write(f"Predicted price: {y_pred[0]:.2f} €")