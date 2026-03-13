from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data = {
"age":[20,25,30],
"salary":[25000,30000,35000]
}

df = pd.DataFrame(data)

# Standardization
scaler = StandardScaler()
standardized = scaler.fit_transform(df)

print("Standardized Data:")
print(standardized)

# Normalization
minmax = MinMaxScaler()
normalized = minmax.fit_transform(df)

print("Normalized Data:")
print(normalized)
