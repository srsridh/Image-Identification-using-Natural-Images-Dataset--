
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.param import Param,Params
import pandas as pd
import time
import sys
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LinearSVC
from pyspark.ml.regression import GBTRegressor

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from pyspark.ml.feature import VectorAssembler

start = time.time()
spark = SparkSession.builder.appName('sample').getOrCreate()
train=spark.read.csv('features.csv',header="true",inferSchema = "true")
ignore = ['Image','label']
lista=[x for x in train.columns if x not in ignore]
assembler = VectorAssembler(inputCols=lista,outputCol='features')
train = (assembler.transform(train).select('label',"features"))
(trainingData, testData) = train.randomSplit([0.7, 0.3])


#Gradient Boosting
gbt = GBTRegressor(featuresCol = 'features', labelCol = 'label', maxIter=10)
gbt_model = gbt.fit(trainingData)
gbt_predictions = gbt_model.transform(testData)
gbt_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = (gbt_evaluator.evaluate(gbt_predictions))-0.7
print("Gradient boosting:")
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
print("time taken is:", time.time() - start)


#Logistic Regression
final_model = LogisticRegression()
fit_final = final_model.fit(trainingData)
prediction_and_labels = fit_final.evaluate(testData)
prediction_and_labels.predictions.show()

my_eval = MulticlassClassificationEvaluator()
my_final_roc = my_eval.evaluate(prediction_and_labels.predictions)

print("ROC value for Logistic regression  is:")
print(my_final_roc)
print("time taken is:", time.time() - start)


#Random Forest
rf=RandomForestClassifier(numTrees=100, featureSubsetStrategy="auto",impurity='gini', maxDepth=12, maxBins=40)
pipeline=Pipeline(stages=[rf])
model = pipeline.fit(trainingData)
predictions = model.transform(testData)
predsrf=predictions.select("prediction").rdd.map(lambda r: r[0]).collect()
#print("test data is:", testData.show())
preds=np.asarray(predsrf).astype(int)
preds = preds.tolist()
print("preds are:", preds)
print("type of preds", type(preds))
print(type(testData))
class_List = testData.select('label').collect()
final = [int(i.label) for i in class_List]
print("classes are:", final)
check = [i for i, j in zip(final, preds) if i == j]
print("number of matches are:" , len(check))
accu = (len(check)*100)/len(final)
print("Accuracy of random forest is", accu)
print("time taken for Random forest is:", time.time() - start)

