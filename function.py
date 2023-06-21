import pandas as pd

def load_true_labels():
    df = pd.read_csv('diabetes.csv')  # Ganti dengan lokasi dan nama file CSV Anda
    true_labels = df['Outcome'].values  # Ganti 'target' dengan nama kolom label di file CSV Anda
    return true_labels