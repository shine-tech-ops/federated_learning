FROM python:3.12-slim
ENV TZ=Asia/Shanghai
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHON_HOME=/usr/local/bin/python
ENV PATH="/usr/lib/postgresql/16/bin:$PATH"
LABEL version="python3.12.3"
WORKDIR /code
COPY regional/ /code/
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources
RUN echo "deb https://mirrors.ustc.edu.cn/debian stable main contrib non-free non-free-firmware" >> /etc/apt/sources.list
RUN echo "deb http://mirrors.ustc.edu.cn/debian stable-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list
RUN apt update && apt install -y git gcc libsnmp-dev snmp-mibs-downloader gnupg lsb-release libpq-dev
RUN pip install -r requirements.txt

