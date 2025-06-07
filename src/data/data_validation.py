import pandas as pd
import numpy as np
import os


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def columns_transformation(df):
  #  df = df.drop('_id', axis = 1)
    df['ChestPainType'] = df['ChestPainType'].replace({
    "ASY": 4,
    "NAP":3,
    "ATA":2,
    "TA":1})

    df['ChestPainType'] = df['ChestPainType'].astype(int)

    df['RestingECG'] = df['RestingECG'].replace({
    "Normal": 1,
    "ST": 2,
    "LVH":3})
    
    df['RestingECG'] = df['RestingECG'].astype(int)

    df['ST_Slope'] = df['ST_Slope'].replace({
    "Flat": 1,
    "Up": 2,
    "Down": 3})
    
    df['ST_Slope'] = df['ST_Slope'].astype(int)
    return df

def save_data(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False)



def main():
    raw_data_path = "./data/raw"
    processed_data_path = "./data/processed"

    train_data = load_data(os.path.join(raw_data_path, "train.csv"))
    test_data = load_data(os.path.join(raw_data_path, "test.csv"))

    train_processing_data = columns_transformation(train_data)
    test_processing_data = columns_transformation(test_data)

    os.makedirs(processed_data_path)

    save_data(train_processing_data, os.path.join(processed_data_path, "tain_processed.csv"))
    save_data(test_processing_data, os.path.join(processed_data_path, "test_processed.csv"))

if __name__ == "__main__":
    main()
