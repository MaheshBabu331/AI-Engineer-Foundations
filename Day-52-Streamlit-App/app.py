import ipywidgets as widgets
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Train model
X, y = load_iris(return_X_y=True)
model = DecisionTreeClassifier()
model.fit(X, y)

# Create input fields
sepal_length = widgets.FloatText(description='Sepal Length')
sepal_width = widgets.FloatText(description='Sepal Width')
petal_length = widgets.FloatText(description='Petal Length')
petal_width = widgets.FloatText(description='Petal Width')

button = widgets.Button(description="Predict")
output = widgets.Output()

# Define prediction function
def predict(b):
    with output:
        output.clear_output()
        data = [[sepal_length.value, sepal_width.value, petal_length.value, petal_width.value]]
        result = model.predict(data)
        print("Prediction:", result[0])

button.on_click(predict)

# Display UI
display(sepal_length, sepal_width, petal_length, petal_width, button, output)
