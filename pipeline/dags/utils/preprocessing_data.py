import os

import numpy as np
import pandas as pd
from loguru import logger
from utils.config import PROCESSED_DATA_FOLDER, SPLIT_RANDOM_STATE, SPLIT_TEST_SIZE, RAW_DATA_FOLDER, \
    PRECIPITACIONES_CSV, BANCO_CENTRAL_CSV, PRECIO_LECHE_CSV

os.environ['LANGUAGE'] = "es_ES.UTF-8"
os.environ['LC_ALL'] = "es_ES.UTF-8"

import locale

locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')

from sklearn.model_selection import train_test_split

from utils.config import SEED


def convert_int(x):
    return int(x.replace('.', ''))


def to_100(x):  # mirando datos del bc, pib existe entre ~85-120 - igual esto es cm (?)
    x = x.split('.')
    if x[0].startswith('1'):  # es 100+
        if len(x[0]) > 2:
            return float(x[0] + '.' + x[1])
        else:
            x = x[0] + x[1]
            return float(x[0:3] + '.' + x[3:])
    else:
        if len(x[0]) > 2:
            return float(x[0][0:2] + '.' + x[0][-1])
        else:
            x = x[0] + x[1]
            return float(x[0:2] + '.' + x[2:])


def preprocessing_data():
    precipitaciones_csv_path = os.path.join(RAW_DATA_FOLDER, PRECIPITACIONES_CSV)
    logger.info(f"Reading Precipitaciones from: {precipitaciones_csv_path}")
    precipitaciones = pd.read_csv(precipitaciones_csv_path)  # [mm]
    precipitaciones['date'] = pd.to_datetime(precipitaciones['date'], format='%Y-%m-%d')
    precipitaciones = precipitaciones.sort_values(by='date', ascending=True).reset_index(drop=True)
    precipitaciones['mes'] = precipitaciones.date.apply(lambda x: x.month)
    precipitaciones['ano'] = precipitaciones.date.apply(lambda x: x.year)
    logger.info(f"Precipitaciones columns: {precipitaciones.columns}")
    logger.info(f"Precipitaciones shape: {precipitaciones.shape}")

    # Banco Central
    banco_central_csv_path = os.path.join(RAW_DATA_FOLDER, BANCO_CENTRAL_CSV)
    logger.info(f"Reading Banco Central from: {banco_central_csv_path}")
    banco_central = pd.read_csv(banco_central_csv_path)
    banco_central['Periodo'] = banco_central['Periodo'].apply(lambda x: x[0:10])
    banco_central['Periodo'] = pd.to_datetime(banco_central['Periodo'], format='%Y-%m-%d', errors='coerce')
    banco_central[banco_central.duplicated(subset='Periodo', keep=False)]  # repetido se elimina
    banco_central.drop_duplicates(subset='Periodo', inplace=True)
    banco_central = banco_central[~banco_central.Periodo.isna()]
    cols_pib = [x for x in list(banco_central.columns) if 'PIB' in x]
    cols_pib.extend(['Periodo'])
    banco_central_pib = banco_central[cols_pib]
    banco_central_pib = banco_central_pib.dropna(how='any', axis=0)
    for col in cols_pib:
        if col == 'Periodo':
            continue
        else:
            banco_central_pib[col] = banco_central_pib[col].apply(lambda x: convert_int(x))
    banco_central_pib.sort_values(by='Periodo', ascending=True)
    cols_imacec = [x for x in list(banco_central.columns) if 'Imacec' in x]
    cols_imacec.extend(['Periodo'])
    banco_central_imacec = banco_central[cols_imacec]
    banco_central_imacec = banco_central_imacec.dropna(how='any', axis=0)

    for col in cols_imacec:
        if col == 'Periodo':
            continue
        else:
            banco_central_imacec[col] = banco_central_imacec[col].apply(lambda x: to_100(x))
            assert (banco_central_imacec[col].max() > 100)
            assert (banco_central_imacec[col].min() > 30)

    banco_central_imacec.sort_values(by='Periodo', ascending=True)
    banco_central_iv = banco_central[['Indice_de_ventas_comercio_real_no_durables_IVCM', 'Periodo']]
    banco_central_iv = banco_central_iv.dropna()  # -unidades? #parte
    banco_central_iv = banco_central_iv.sort_values(by='Periodo', ascending=True)
    banco_central_iv['num'] = banco_central_iv.Indice_de_ventas_comercio_real_no_durables_IVCM.apply(
        lambda x: to_100(x))
    banco_central_num = pd.merge(banco_central_pib, banco_central_imacec, on='Periodo', how='inner')
    banco_central_num = pd.merge(banco_central_num, banco_central_iv, on='Periodo', how='inner')
    banco_central_num['mes'] = banco_central_num['Periodo'].apply(lambda x: x.month)
    banco_central_num['ano'] = banco_central_num['Periodo'].apply(lambda x: x.year)
    logger.info(f"Banco Central columns: {banco_central_num.columns}")
    logger.info(f"Banco Central shape: {banco_central_num.shape}")

    # Precio Leche
    precio_leche_csv_path = os.path.join(RAW_DATA_FOLDER, PRECIO_LECHE_CSV)
    logger.info(f"Reading Precio Leche from: {precio_leche_csv_path}")
    precio_leche = pd.read_csv(precio_leche_csv_path)
    precio_leche.rename(columns={'Anio': 'ano', 'Mes': 'mes_pal'},
                        inplace=True)  # precio = nominal, sin iva en clp/litro
    precio_leche['mes'] = pd.to_datetime(precio_leche['mes_pal'], format='%b')
    precio_leche['mes'] = precio_leche['mes'].apply(lambda x: x.month)
    precio_leche['mes-ano'] = precio_leche.apply(lambda x: f'{x.mes}-{x.ano}', axis=1)
    logger.info(f"Precio Leche columns: {banco_central_num.columns}")
    logger.info(f"Precio Leche shape: {banco_central_num.shape}")

    # Merge Data
    logger.info(f"Merge Data")
    precio_leche_pp = pd.merge(precio_leche, precipitaciones, on=['mes', 'ano'], how='inner')
    precio_leche_pp.drop('date', axis=1, inplace=True)
    precio_leche_pp_pib = pd.merge(precio_leche_pp, banco_central_num, on=['mes', 'ano'], how='inner')
    precio_leche_pp_pib.drop(['Periodo', 'Indice_de_ventas_comercio_real_no_durables_IVCM', 'mes-ano', 'mes_pal'],
                             axis=1, inplace=True)

    X = precio_leche_pp_pib.drop(['Precio_leche'], axis=1)
    y = precio_leche_pp_pib['Precio_leche']
    logger.info(f"X data columns: {X.columns}. X data shape: {X.shape}")

    # generate random data-set
    logger.info(f"Split Dataset")
    np.random.seed(SEED)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT_TEST_SIZE,
                                                        random_state=SPLIT_RANDOM_STATE)
    logger.info(f"Train Examples: {len(X_train)}")
    logger.info(f"Test Examples: {len(X_test)}")

    logger.info(f"Saving data (X_train.csv, y_train.csv, X_test.csv, y_test.csv) to {PROCESSED_DATA_FOLDER}")
    X_train.to_csv(os.path.join(PROCESSED_DATA_FOLDER, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DATA_FOLDER, 'y_train.csv'), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DATA_FOLDER, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DATA_FOLDER, 'y_test.csv'), index=False)
    logger.info(f"Processing done")
