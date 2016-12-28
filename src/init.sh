#!/bin/bash

# download files from bucket
gsutil cp gs://topic-zoomer/requirements.txt .
# install deps
sudo apt-get install -y python3-pip
sudo pip3 install -r requirements.txt
# setup python3 support
# see https://blog.sourced.tech/post/dataproc_jupyter/
echo "export PYSPARK_PYTHON=python3" | tee -a  /etc/profile.d/spark_config.sh  /etc/*bashrc /usr/lib/spark/conf/spark-env.sh
echo "export PYTHONHASHSEED=0" | tee -a /etc/profile.d/spark_config.sh /etc/*bashrc /usr/lib/spark/conf/spark-env.sh
echo "spark.executorEnv.PYTHONHASHSEED=0" >> /etc/spark/conf/spark-defaults.conf
