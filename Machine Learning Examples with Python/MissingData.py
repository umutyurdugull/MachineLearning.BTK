import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

datas = pd.read_csv('eksikveriler.csv')
print(datas)

#missing datas

imputer = SimpleImputer(missing_values=np.nan,strategy="mean")

age = datas.iloc[:,1:4].values
imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])
print(age)