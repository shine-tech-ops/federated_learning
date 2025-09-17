FROM python:slim
ENV TZ=Asia/Shanghai
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHON_HOME=/usr/local/bin/python
ENV PATH="/usr/lib/postgresql/16/bin:$PATH"
LABEL version="python3.12.3"
WORKDIR /code
COPY backend/ /code/
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources
RUN echo "deb https://mirrors.ustc.edu.cn/debian stable main contrib non-free non-free-firmware" >> /etc/apt/sources.list
RUN echo "deb http://mirrors.ustc.edu.cn/debian stable-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list
RUN apt update && apt install -y git gcc libsnmp-dev snmp-mibs-downloader gnupg lsb-release libpq-dev
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/keyrings/postgresql-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/postgresql-keyring.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" | tee /etc/apt/sources.list.d/pgdg.list
RUN apt update && apt install postgresql-client-16 -y
RUN pip install -r requirements.txt
