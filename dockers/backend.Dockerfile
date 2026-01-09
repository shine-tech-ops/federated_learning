FROM python:slim
ENV TZ=Asia/Shanghai
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHON_HOME=/usr/local/bin/python
ENV PATH="/usr/lib/postgresql/16/bin:$PATH"
LABEL version="python3.12.3"
WORKDIR /code
COPY backend/ /code/
RUN apt-get update \
    && apt-get install -y wget git gcc libsnmp-dev gnupg lsb-release libpq-dev postgresql-client \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
