import os
import yaml

CONFIG_PATH = "config.yml"
DBT_DIR = "dbt"
DBT_PROJECT_PATH = os.path.join(DBT_DIR, "dbt_project.yml")
DBT_PROFILES_PATH = os.path.join(DBT_DIR, "profiles.yml")
MODELS_DIR = os.path.join(DBT_DIR, "models", "ingestion")


def read_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def setup_dbt_files(config):
    os.makedirs(DBT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # dbt_project.yml
    dbt_project_data = {
        "name": "acd_map",
        "version": "1.0",
        "profile": "acd_map",
        "model-paths": ["models"],
        "target-path": "target",
        "clean-targets": ["target"],
    }
    write_yaml(DBT_PROJECT_PATH, dbt_project_data)

    # profiles.yml
    database_name = config.get("database", {}).get("name", "default_db")
    profiles_data = {
        "acd_map": {
            "target": "dev",
            "outputs": {"dev": {"type": "duckdb", "path": f"{database_name}.duckdb"}},
        }
    }
    write_yaml(DBT_PROFILES_PATH, profiles_data)

    # Create .sql files for each model (using DuckDB's read_csv_auto for csv directories)
    models = config.get("models", [])
    for model in models:
        model_name = model.get("name")
        data_folder = model.get("data_folder")
        if model_name and data_folder:
            sql_path = os.path.join(MODELS_DIR, f"{model_name}.sql")
            with open(sql_path, "w") as sql_file:
                sql_file.write(f"-- Model: {model_name}\n")
                sql_file.write(f"-- Source: {data_folder}\n\n")
                # sensors = list of sensor names
                sensors = model.get("sensors", [])
                columns = [
                    "try_strptime(time, ['%Y/%m/%d %H:%M:%S:%f', '%Y/%m/%d %H:%M:%S', '%Y-%m-%d_%H-%M-%S']) as ts",
                    "date_part('month', ts) as month",
                    "date_part('day', ts) as day",
                    "date_part('hour', ts) as hour",
                ] + [f'"{s}"' for s in sensors]
                selection = "SELECT " + ", ".join(columns)
                sql_file.write(
                    f"{selection}\nFROM read_csv_auto('{data_folder}/*.csv')\n"
                )


if __name__ == "__main__":
    config = read_config(CONFIG_PATH)
    try:
        setup_dbt_files(config)
        print("DBT project (and models) initialized successfully based on config.yml")
    except Exception as e:
        print(f"Error initializing DBT project based on config.yml: {e}")
