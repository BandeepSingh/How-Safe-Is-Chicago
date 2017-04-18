__author__ = 'uzmaa'
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF, IDF
import sys, operator
import string
import math
from pyspark.mllib.regression import *#LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import Normalizer


def xRMSerror (parsedDataTrain,parsedDataTest):

     numIterations = 1000
     stepsize=6

     model = LinearRegressionWithSGD.train(parsedDataTrain,numIterations,stepsize)


     # Evaluate the model on training data
     valuesAndPreds = parsedDataTest.map(lambda p: (p.label, model.predict(p.features)))
     print valuesAndPreds.take(5)
     MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
     return math.sqrt(MSE)


def parseDataF(x):

     label=0

     features=x[1]
     if x[0][1]!="None" and x[0][1] is not None:
          label=int(x[0][1])
     return LabeledPoint(int(label),features)


conf = SparkConf().setAppName('Chicago Linear Reg')
sc = SparkContext(conf=conf)

normalizer1 = Normalizer(1)

v = Vectors.dense(range(3))
nor = Normalizer(1)
normalizer1.transform(v)
print normalizer1.transform(v)

inputs =  sys.argv[1]
#inputslabel = "/Volumes/personal/uzmaa-irmacs/Chicago/data/2015CrimeCounts.txt"FeatureSetByCrimePickle

inputrdd=sc.pickleFile(inputs)

inputrddidlabel=inputrdd.map(lambda (id,(features,label)):(id,label))

inputrddidfeatures=inputrdd.map(lambda (id,(features,label)):features)

inputrddNorm=inputrddidlabel.zip(normalizer1.transform(inputrddidfeatures))

print inputrddNorm.take(100)

inputrddNormData=inputrddNorm.filter(lambda (((Community,Year,Month),label),features):Year<2015 and label!="None" and label is not None)

inputrddtrain, inputrddtest =  inputrddNormData.randomSplit([10,4])

inputrddval=inputrddNorm.filter(lambda (((Community,Year,Month),label),features):Year>=2015)

DataTrain = inputrddtrain.map(parseDataF)
DataTest = inputrddtest.map(parseDataF)

numIterations = 100000
stepsize=0

model = LinearRegressionWithSGD.train(DataTrain,numIterations)

# Evaluate the model on training data
valuesAndPreds = DataTrain.map(lambda p: (p.label, model.predict(p.features)))
MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()

print("Best Test Root Mean Squared Error: " + str(math.sqrt(MSE)))
