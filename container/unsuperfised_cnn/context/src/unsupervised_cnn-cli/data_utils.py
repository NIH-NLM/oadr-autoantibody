# data_utils.py
import pandas as pd

def load_and_prepare_data(input_file):
    df = pd.read_csv(input_file)
    df = df[['Accession', 'Property', 'Value', 'Sex', 'Age_Group','Study']]

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['Sex', 'Age_Group', 'Property','Study'])

    # Pivot to matrix: rows = Accessions, columns = features
    df_matrix = df.groupby('Accession').mean().fillna(0)

    return df_matrix
