FROM ubuntu:latest

MAINTAINER Sergey Kulakin 'sergeykulakin@gmail.com'

# install requared packages
RUN apt-get update -y && apt-get install -y python3-pip python3-dev build-essential

# copy folder with all service files
COPY . /app

# change current dir
WORKDIR /app

# isntall required python modules
RUN pip install --upgrade pip; \
    pip install -r requirements.txt

# expose desired port
EXPOSE 5000

# run python script with provided args
CMD ["python3", "app.py", "--host", "0.0.0.0", "--port", "5000"]