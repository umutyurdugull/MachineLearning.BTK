import nltk
from nltk.tokenize import word_tokenize
import re
import pandas as pd


nltk.download("punkt")

text = "Bunlar gerçek hayatta ne işimize yarayacak?"
tokens = text.split()
print(tokens)

tokens = word_tokenize(text)
print(tokens)

tokens = re.findall(r'\w+',text)
print(tokens)

df = pd.DataFrame({"text": [text]})
df['tokens'] = df['text'].str.split()
print(df['tokens'][0])
