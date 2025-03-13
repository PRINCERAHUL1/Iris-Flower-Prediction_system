import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Load the model
model = pickle.load(open('iris_flower_prediction_model.sav', 'rb'))

st.title('🌺 Iris Flower Prediction System')

# Get user input
st.subheader('🌷 Enter Flower Data')

def get_flower_input():
    sl = st.number_input("🌿 Sepal Length (cm)", min_value=0.0, step=0.1)
    sw = st.number_input("🌿 Sepal Width (cm)", min_value=0.0, step=0.1)
    pl = st.number_input("🌸 Petal Length (cm)", min_value=0.0, step=0.1)
    pw = st.number_input("🌸 Petal Width (cm)", min_value=0.0, step=0.1)

    # Create DataFrame for input
    flower_data = pd.DataFrame([[sl, sw, pl, pw]], 
                               columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
    return flower_data

# Get input data
flower_input = get_flower_input()

# Display user input
st.subheader('📊 Flower Data Overview')
st.write(flower_input)

# Button to predict
if st.button("🔍 Predict"):
    
    prediction = model.predict(np.array(flower_input))

    # Display prediction
    st.markdown(f'<p style="font-size: 24px; font-weight: bold;">🌼 Predicted Flower: {prediction[0]}</p>', unsafe_allow_html=True)


    # Display corresponding image
    if prediction[0] == 'Setosa':
        st.image("Setosa.jpg", caption="Iris Setosa")
    elif prediction[0]=='Versicolor':
        st.image("Versicolor.jpg", caption="Iris Versicolor")
    elif prediction[0]=='Virginica':
        st.image("Virginica.jpg", caption="Iris Virginica")