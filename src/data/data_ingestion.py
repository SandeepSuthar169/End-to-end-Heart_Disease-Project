import pandas as pd
import numpy as np
import yaml
import os
from pymongo import MongoClient
from sklearn.model_selection import train_test_split


def load_params(filepath: str) -> float:
    with open(filepath, 'r') as file:
        params = yaml.safe_load(file)
        return params['data_ingestion']['test_size'] 
    

def load_data(filepath: str ) -> pd.DataFrame:
    return pd.read_csv(filepath)

def split_data(df: pd.DataFrame, test_size = float):
    return train_test_split(df, test_size=test_size, random_state=42)


def save_data(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index = False)


def upload_csv_to_mongodb(df: pd.DataFrame, db_name: str, collection_name: str, connection_url: str) -> None:
    clinet = MongoClient(connection_url)
    db = clinet[db_name]
    collection = db[collection_name]
    data = df.to_dict(orient='records')
    collection.insert_many(data)

def main():
    data_filepath = r"C:/Users/Sande/Desktop/project_2/notebook/heart.csv"
    params_filepath = 'params.yaml'
    raw_data_path = os.path.join('data', 'raw')

    os.makedirs(raw_data_path, exist_ok=True)

    df = load_data(data_filepath)
    test_size = load_params(params_filepath)

    train_data, test_data = split_data(df, test_size)
    
    save_data(train_data, os.path.join(raw_data_path, "train.csv"))
    save_data(test_data, os.path.join(raw_data_path, "test.csv"))

    CONNECTION_URL = "xxxxxx/xxxxxxxxxxxx/xxxxxxxxxxxxxxxx/xxxxxxxxxxxxxxxxxx/xxxxxxxxxxxxxxx" # this string you provided is a MongoDB connection URI
    DB_NAME = "heart"
    COLLENTION_NAME = "heart-Data"

    upload_csv_to_mongodb(df, DB_NAME, COLLENTION_NAME, CONNECTION_URL)


if __name__ == "__main__":
    main()