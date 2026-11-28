#libraries 
import numpy as np
import matplotlib as plt
import pandas as pd

#import data 

veriler = pd.read_csv('veriler.csv')
print(veriler)
print(veriler[['boy','kilo']])



#data preprocessing
