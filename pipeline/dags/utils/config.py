EXP_CONFIG = {"exp_name": "exp-01",
              "k": [3, 4, 5, 6, 7, 10],
              "alpha": [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
              "poly": [1, 2, 3, 5, 7],
              'seed': 0,
              "random_state": 42,
              "split_test_size": 0.2}

RAW_DATA_FOLDER = 'data/raw_data'
PROCESSED_DATA_FOLDER = 'data/processed_data'
ABS_PATH = '/opt/airflow/'
MODELS_PATH = 'models'

# DATA URL. Just to not write them every time
BANCO_CENTRAL_DATA_URL = "https://raw.githubusercontent.com/SpikeLab-CL/ml-engineer-challenge/main/data_scientist_past_challenge/data/banco_central.csv"
PRECIPITACIONES_DATA_URL = "https://raw.githubusercontent.com/SpikeLab-CL/ml-engineer-challenge/main/data_scientist_past_challenge/data/precipitaciones.csv"
PRECIO_LECHE_DATA_URL = "https://raw.githubusercontent.com/SpikeLab-CL/ml-engineer-challenge/main/data_scientist_past_challenge/data/precio_leche.csv"

BANCO_CENTRAL_CSV = 'banco_central.csv'
PRECIPITACIONES_CSV = 'precipitaciones.csv'
PRECIO_LECHE_CSV = 'precio_leche.csv'
