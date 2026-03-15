import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sample dataset
data = {
    "City": ["Delhi", "Mumbai", "Chennai", "Delhi"],
    "Salary": [50000, 60000, 55000, 52000]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Label Encoding
encoder = LabelEncoder()
df["City_Label"] = encoder.fit_transform(df["City"])

print("\nLabel Encoded Data:")
print(df)

# One-Hot Encoding
one_hot = pd.get_dummies(df["City"], prefix="City")

df_one_hot = pd.concat([df, one_hot], axis=1)

print("\nOne-Hot Encoded Data:")
print(df_one_hot)
