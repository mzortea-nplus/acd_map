## ACD–MAP Analysis Pipeline

This project reads accelerometer data from CSVs via DuckDB + dbt, then runs analysis and plotting in Python.

### 1. Download / clone the repository

- **Clone the repo** (or download and unzip) into a local folder, e.g.:

```bash
git clone <your_repo_url> acd_map
cd acd_map
```

Make sure you are in the project root (the folder containing `config.yml`, `setup_dbt.py`, and `main.py`).

### 2. Configure `config.yml`

- **Open** `config.yml` and set:
  - **`models`**:
    - `name`: logical name of each device/table (e.g. `acd`, `map`).
    - `data_folder`: absolute path to the folder containing that device’s CSV files.
    - `sensors`: list of column names (as strings) to analyse (e.g. `"03056B18_x"`).
  - **`database.name`**: DuckDB file name to create/use (e.g. `mydb`).
  - **`analysis` flags** (if present): enable/disable specific analyses and plots.

Save the file before continuing.

### 3. Initialize the dbt project from the config

From the project root:

```bash
python setup_dbt.py
```

This will:
- Create/update the `dbt/` project (`dbt_project.yml`, `profiles.yml`).
- Generate one `.sql` model per entry in `config.yml` under `models`, each reading the corresponding CSV files via DuckDB.

### 4. Build the DuckDB models with dbt

Change into the `dbt/` directory and run dbt:

```bash
cd dbt
dbt run
```

This will materialize the views/tables (e.g. `acd`, `map`) in the DuckDB database specified in `profiles.yml` / `config.yml`.

### 5. Run the main analysis script

Return to the project root and call `main.py`, passing the DuckDB file path created in step 4 (usually in `dbt/` and named `<database.name>.duckdb`):

```bash
cd ..
python main.py dbt/<your_database_name>.duckdb
```

For example, if `database.name: mydb` in `config.yml`:

```bash
python main.py dbt/mydb.duckdb
```

The script will:
- Read the dbt models (e.g. `acd`, `map`) from DuckDB.
- Use the sensors defined in `config.yml`.
- Generate figures under `figures/` (time plots, offsets, spectra, sine fits, etc.) and a `results.csv` summary if those analyses are enabled.

You can then inspect the generated plots and `results.csv` to interpret the offsets and spectral/phase characteristics between devices.

