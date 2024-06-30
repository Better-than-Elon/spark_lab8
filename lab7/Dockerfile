FROM python:3.9-slim
COPY --from=openjdk:8-jre-slim /usr/local/openjdk-8 /usr/local/openjdk-8

ENV JAVA_HOME /usr/local/openjdk-8

RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk-8/bin/java 1

WORKDIR /app

ADD ./requirements.txt /app/requirements.txt

RUN pip3 install -r requirements.txt

ADD . /app

#CMD ["python", "kmeans.py"]

# Set the entrypoint
#CMD ["spark-submit", "--master", "local[*]", "pysparkU.py"]