#!/bin/bash

docker build -t ridge-api app/
docker run -d -p 80:80 -v $PWD/models:/app/models -v $PWD/app/logs:/app/logs --name ridge-api-container ridge-api
