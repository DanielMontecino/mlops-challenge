import io
import os

import pandas as pd
import requests
from utils.config import BANCO_CENTRAL_DATA_URL, PRECIPITACIONES_DATA_URL, PRECIO_LECHE_DATA_URL, RAW_DATA_FOLDER, \
    BANCO_CENTRAL_CSV, PRECIPITACIONES_CSV, PRECIO_LECHE_CSV


def download_data(url, name):
    s = requests.get(url).content
    c = pd.read_csv(io.StringIO(s.decode('utf-8')))
    c.to_csv(name, index=False)


def get_banco_central():
    url = BANCO_CENTRAL_DATA_URL
    name = os.path.join(RAW_DATA_FOLDER, BANCO_CENTRAL_CSV)
    download_data(url, name)


def get_precipitaciones():
    url = PRECIPITACIONES_DATA_URL
    name = os.path.join(RAW_DATA_FOLDER, PRECIPITACIONES_CSV)
    download_data(url, name)


def get_precio_leche():
    url = PRECIO_LECHE_DATA_URL
    name = os.path.join(RAW_DATA_FOLDER, PRECIO_LECHE_CSV)
    download_data(url, name)
