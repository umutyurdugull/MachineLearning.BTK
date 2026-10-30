import pandas as pd

data = {
    'Musteri_ID': [1, 2, 3, 4, 5],
    'Yas': [25, 48, 32, 61, 28],
    'Gelir': [45000, 120000, 60000, 95000, 38000],
    'Egitim_Duzeyi': ['Üniversite', 'Yüksek Lisans', 'Lise', 'Doktora', 'Ön Lisans'],
    'Web_Suresi_sn': [180, 60, 300, 90, 240]
}
df = pd.DataFrame(data)
print(df.describe())