import ipywidgets as widgets
from IPython.display import display
import numpy as np
from sklearn.linear_model import LinearRegression

# train model
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([20000, 30000, 40000, 50000, 60000])
model = LinearRegression()
model.fit(X, y)

# input box
exp = widgets.FloatText(description="Experience")
button = widgets.Button(description="Predict")
output = widgets.Output()

def predict(b):
    with output:
        output.clear_output()
        result = model.predict([[exp.value]])
        print("Salary:", int(result[0]))

button.on_click(predict)

display(exp, button, output)
