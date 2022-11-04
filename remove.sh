#!/bin/bash

cd pipeline
docker-compose down --volumes --remove-orphans
docker rm -f ridge-api-container

docker rmi  ridge-api airflow-custom

