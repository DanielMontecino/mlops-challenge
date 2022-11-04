import os
from datetime import datetime

from airflow.models import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from loguru import logger

from utils.config import BANCO_CENTRAL_DATA_URL, PRECIPITACIONES_DATA_URL, PRECIO_LECHE_DATA_URL, RAW_DATA_FOLDER, \
    BANCO_CENTRAL_CSV, PRECIPITACIONES_CSV, PRECIO_LECHE_CSV, ABS_PATH, EXP_CONFIG
from utils.preprocessing_data import preprocessing_data
from utils.test_model import test_model
from utils.train_model import train_model

logger.add('logs/pipeline_logs', level='INFO')

default_args = {
    'owner': 'Daniel Montecino M.',
    'email_on_failure': False,
    'email': ['daniel.montecino@ug.uchile.cl'],
    'start_date': datetime(2022, 11, 1)
}

with DAG(
        "ml_pipeline",
        description='End-to-end ML',
        default_args=default_args,
        catchup=False,
        params=EXP_CONFIG) as dag:
    # task 0
    with TaskGroup('get_data') as get_data:
        # =======
        # task: 0.1
        # banco_central = PythonOperator(
        #     task_id='get_banco_central',
        #     python_callable=get_banco_central
        # )
        #
        # # task: 0.2
        # precipitaciones = PythonOperator(
        #     task_id='get_precipitaciones',
        #     python_callable=get_precipitaciones
        # )
        #
        # # task: 0.3
        # precio_leche = PythonOperator(
        #     task_id='get_precio_leche',
        #     python_callable=get_precio_leche
        # )

        banco_central = BashOperator(
            task_id='get_banco_central',
            bash_command=f'wget -O {os.path.join(ABS_PATH, RAW_DATA_FOLDER, BANCO_CENTRAL_CSV)} {BANCO_CENTRAL_DATA_URL}'
        )

        # task: 0.2
        precipitaciones = BashOperator(
            task_id='get_precipitaciones',
            bash_command=f'wget -O {os.path.join(ABS_PATH, RAW_DATA_FOLDER, PRECIPITACIONES_CSV)} {PRECIPITACIONES_DATA_URL}'
        )

        # task: 0.3
        precio_leche = BashOperator(
            task_id='get_precio_leche',
            bash_command=f'wget -O {os.path.join(ABS_PATH, RAW_DATA_FOLDER, PRECIO_LECHE_CSV)} {PRECIO_LECHE_DATA_URL}'
        )

    # task: 1
    preprocessing = PythonOperator(
        task_id='preprocessing',
        python_callable=preprocessing_data

    )

    # task: 2
    train = PythonOperator(
        task_id='train',
        provide_context=True,
        python_callable=train_model
    )

    # task: 3
    test = PythonOperator(
        task_id='test',
        python_callable=test_model
    )

    get_data >> preprocessing >> train >> test
