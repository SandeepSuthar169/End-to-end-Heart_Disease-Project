import os
from  pathlib import Path

project_name = "src"

list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/data/__init__.py",
    f"{project_name}/data/data_ingestion.py",
    f"{project_name}/data/data_validation.py",
    f"{project_name}/features/__init__.py",
    f"{project_name}/features/data_transformation.py",
    f"{project_name}/model/__init__.py",
    f"{project_name}/model/model_building.py",
    f"{project_name}/model/model_evaluation.py",
    f"{project_name}/model/model_pusher.py",
    f"{project_name}/visualization/__init__.py",
    f"{project_name}/visualization/visualize.py",
    "app.py",
    "config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "demo.py",
    "setup.py",
    "config/model.yaml",
    "config/schema.yaml"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as file:
            pass
    else:
        print(f"file is alreay present at : {filepath}")          