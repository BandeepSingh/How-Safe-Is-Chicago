__author__ = 'uzmaa'
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF, IDF
import sys, operator
import string
import math
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import Normalizer


def xRMSerror (parsedDataTrain,parsedDataTest):

     numIterations = 1000
     stepsize=0

     model = LinearRegressionWithSGD.train(parsedDataTrain,numIterations,stepsize)


     # Evaluate the model on training data
     valuesAndPreds = parsedDataTest.map(lambda p: (p.label, model.predict(p.features)))
     print valuesAndPreds.take(5)
     MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
     return math.sqrt(MSE)


def parseDataF(x):

     label=0
     features=x[1][0]
     if x[1][1]!="None" and x[1][1] is not None:
          label=int(x[1][1])
     return LabeledPoint(int(label),features)



conf = SparkConf().setAppName('Chicago Linear Reg')
sc = SparkContext(conf=conf)

normalizer1 = Normalizer()
inputs =  sys.argv[1]#"/Volumes/personal/uzmaa-irmacs/Chicago/data/FeatureSetSocialCrimePickle"

inputrdd=sc.pickleFile(inputs)

inputrddtrain=inputrdd.filter(lambda ((Community,Year,Month),(features,label)):Year<2010)

inputrddtest=inputrdd.filter(lambda ((Community,Year,Month),(features,label)):Year<2015 and Year>=2010)

DataTrain = inputrddtrain.map(parseDataF)
DataTest = inputrddtest.map(parseDataF)

print DataTrain.take(5)
print DataTest.take(5)


RMSE = xRMSerror(DataTrain,DataTest)

print("Best Test Root Mean Squared Error: " + str(RMSE))# + " for Step Size " + str(optStepsize))
