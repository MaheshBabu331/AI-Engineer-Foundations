import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# train model
X, y = load_iris(return_X_y=True)
model = DecisionTreeClassifier()
model.fit(X, y)

st.title("Iris Prediction App")

# user inputs
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# button
if st.button("Predict"):
    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.write("Prediction:", result[0])
