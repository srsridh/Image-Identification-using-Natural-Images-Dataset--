#!/bin/bash

numExecutor=2
coresPerWorker=1
memExecutor=4G

data=/N/u/srsridh/natural-images/features.csv



spark-submit --deploy-mode client --num-executors ${numExecutor} --executor-cores ${coresPerWorker} --executor-memory ${memExecutor} ./Image_Classification_Spark.py ${data}