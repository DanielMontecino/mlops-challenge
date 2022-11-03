FROM apache/airflow:2.2.2
User root
RUN apt update && apt install -y --no-install-recommends locales; rm -rf /var/lib/apt/lists/*; sed -i '/^#.* es_ES.UTF-8 /s/^#//' /etc/locale.gen; locale-gen
RUN apt-get update && apt-get install wget -y
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir loguru
ENV LANG es_ES.UTF-8
ENV LC_NUMERIC es_ES.UTF-8
