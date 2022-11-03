#!/bin/bash

mkdir -p ./pipeline/logs ./pipeline/plugins ./data/raw_data ./data/processed_data ./models
echo -e "AIRFLOW_UID=$(id -u)" > ./pipeline/.env

cd pipeline
docker build -t airflow-custom .
docker-compose up airflow-init
docker-compose up