import pandas as pd
import numpy as np
import os
import yaml
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def load_params(params_path: str):
    with open(params_path, 'r') as file:
        params  = yaml.safe_load(file)
        n_estimators = params['model_building']['n_estimators']
        max_depth = params['model_building']['max_depth']
        return n_estimators, max_depth

def load_data(filepath:str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def perpare_features(df: pd.DataFrame):
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']
    return X, y

def build_pipeline(n_estimators: int,
                   max_depth: int) -> Pipeline:
    process = ColumnTransformer(transformers=[
    ('one', OneHotEncoder(), ['ExerciseAngina', 'Sex']),
    ('std', StandardScaler(), ['MaxHR', 'Age', 'RestingBP', 'Cholesterol'])
    ],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('proprocess', process),
        ('classi', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth))
    ])
    
    return pipeline



def train(pipeline:Pipeline, X, y) -> Pipeline:
    pipeline.fit(X, y)
    return pipeline

def save_model(pipeline: Pipeline, output_path: str):
    with open(output_path, 'wb') as file:
        pickle.dump(pipeline, file)

def main():
    input_Path = "./data/processed/tain_processed.csv"
    model_path = "model.pkl"
    params_path = "params.yaml"

    n_estimators, max_depth = load_params(params_path)
    df = load_data(input_Path)
    X_train, y_train = perpare_features(df)

    Pipeline = build_pipeline(n_estimators, max_depth)

    trained_pipeline = train(Pipeline, X_train, y_train)

    save_model(trained_pipeline, model_path)


if __name__ == "__main__":
    main()
