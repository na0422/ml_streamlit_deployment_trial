import streamlit as st
import joblib
import numpy as np

# 1. Load the model from the same directory
model = joblib.load('iris_model.joblib')

# 2. Build the user interface
st.title("🌸 Iris Flower Predictor")
st.write("Adjust the sliders to predict the flower species.")

# Create input sliders
sepal_length = st.slider('Sepal Length', 4.0, 8.0, 5.8)
sepal_width = st.slider('Sepal Width', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length', 1.0, 7.0, 4.3)
petal_width = st.slider('Petal Width', 0.1, 2.5, 1.3)

# 3. Add a prediction button
if st.button("Predict Species"):
    # Format the inputs for the model
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make the prediction
    prediction = model.predict(features)[0]
    species_names = ['Setosa', 'Versicolor', 'Virginica']
    
    # Display the result
    st.success(f"The predicted species is: **{species_names[prediction]}**")
