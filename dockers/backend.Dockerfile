FROM python:slim
ENV TZ=Asia/Shanghai
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHON_HOME=/usr/local/bin/python
ENV PATH="/usr/lib/postgresql/16/bin:$PATH"
LABEL version="python3.12.3"
WORKDIR /code
COPY backend/ /code/
RUN rm -f /etc/apt/sources.list.d/debian.sources \
    && echo "deb https://mirrors.ustc.edu.cn/debian stable main contrib non-free non-free-firmware" > /etc/apt/sources.list \
    && echo "deb http://mirrors.ustc.edu.cn/debian stable-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y wget git gcc libsnmp-dev snmp-mibs-downloader gnupg lsb-release libpq-dev postgresql-client \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
