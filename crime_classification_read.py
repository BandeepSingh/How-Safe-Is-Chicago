__author__ = 'BANDEEP SINGH'
from pyspark import SparkConf, SparkContext,SQLContext, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType,DateType
from pyspark.sql.functions import *
import sys, operator,re,math
from pyspark.sql.functions import udf
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.sql.types import StringType,FloatType,ArrayType
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.evaluation import MulticlassMetrics

conf = SparkConf().setAppName('How Safe is Chicago')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

label_output = sys.argv[1]
prediction_output = sys.argv[2]

def convert_type(a):
	b=map(int, a)
	return b
	
def label_split(data):
	return data[-1]
	
def feature_split(data):
	return data[0:-1]
	
def parsePoint(line):
	#a=a+line[8:15]
	#b=[float(i) for i in a]
	return LabeledPoint(int(line[0]),line[1:8])
	
process_data_read = sqlContext.read.parquet("output")
process_data_read.registerTempTable('data')
#process_data=sqlContext.sql("""
# SELECT d.label,d.bool_arrest,d.bool_domestic,d.Beat,d.District,d.Ward,d.Community,d.FBICode
# FROM data d where d.label is not null and d.IUCR!='04B' and d.Beat!='04B' and d.District!='04B' and d.Ward!='04B' and d.Community!='04B' and d.FBICode!='04B' and d.IUCR!='04A' and d.Beat!='04A' and d.District!='04A' and d.Ward!='04A' and d.Community!='04A' and d.FBICode!='04A' and d.Ward!='' and d.Community!='' and d.FBICode!='01A' and d.IUCR!='031A' and d.IUCR!='143A' and d.IUCR!='502P' and d.IUCR!='033B' and d.IUCR!='033A'
# """)

process_data=sqlContext.sql("""
 SELECT d.label,d.bool_arrest,d.bool_domestic,d.Beat,d.District,d.Ward,d.Community,d.FBICode
 FROM data d where d.label is not null and d.Beat!='04B' and d.District!='04B' and d.Ward!='04B' and d.Community!='04B' and d.FBICode!='04B' and d.Beat!='04A' and d.District!='04A' and d.Ward!='04A' and d.Community!='04A' and d.FBICode!='04A' and d.Ward!='' and d.Community!='' and d.FBICode!='01A' and d.FBICode!='01B' and d.District!=''
 """)

#print process_data_read.show()
process_rdd= process_data.map(tuple).map(convert_type).map(parsePoint)
(trainingData, testData) = process_rdd.randomSplit([0.7, 0.3])

#model = GradientBoostedTrees.trainRegressor(trainingData,categoricalFeaturesInfo={},numIterations=10)
#model = RandomForest.trainClassifier(trainingData,numClasses=2, categoricalFeaturesInfo={},numTrees=3, featureSubsetStrategy="auto")

#predictions = model.predict(testData.map(lambda p: p.features))
#labelsAndPredictions = testData.map(lambda l: l.label).zip(predictions)
#labelsAndPredictions = testData.map(lambda l: l.label).zip(predictions)
#PredictionsAndlabels = predictions.zip(testData.map(lambda l: l.label))
#metrics = MulticlassMetrics(PredictionsAndlabels)
#print('Precision = ' + str(metrics.precision()))
#print('Recall = ' + str(metrics.recall()))
#testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() /float(testData.count())
#RMSE=math.sqrt(int(testMSE))
#print('Test Error = ' + str(testErr))
#print("Mean Squared Error = " + str(RMSE))

#predictions.coalesce(1).saveAsTextFile(prediction_output)
#output_label=testData.map(lambda l: l.label)
#output_label.coalesce(1).saveAsTextFile(label_output)


model = RandomForest.trainClassifier(trainingData,numClasses=36, categoricalFeaturesInfo={},numTrees=3, featureSubsetStrategy="auto",impurity='gini',)
									 
predictions = model.predict(testData.map(lambda x: x.features))
PredictionsAndlabels = predictions.zip(testData.map(lambda l: l.label))
metrics = MulticlassMetrics(PredictionsAndlabels)
print('Precision = ' + str(metrics.weightedPrecision))
print('Recall = ' + str(metrics.weightedRecall))
#testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
#print('Test Error = ' + str(testErr))
