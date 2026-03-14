import pandas as pd
from sklearn.impute import SimpleImputer

data = {
"Age":[25,30,None,35],
"Salary":[30000,None,45000,40000]
}

df = pd.DataFrame(data)

imputer = SimpleImputer(strategy="mean")

filled_data = imputer.fit_transform(df)

print(filled_data)
