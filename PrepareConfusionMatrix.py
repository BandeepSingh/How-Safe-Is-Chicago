__author__ = 'uzmaa'
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF, IDF
import sys, operator
import string
import math
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import Normalizer
from pyspark.mllib.evaluation import MulticlassMetrics

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

inputs = sys.argv[1] #"/Volumes/personal/uzmaa-irmacs/Chicago/data/FeatureSetByCrimePickleC"
output= sys.argv[2] #"/users/uzmaa/Desktop/output"

inputrdd=sc.pickleFile(inputs)

inputrddidlabel=inputrdd.map(lambda (id,(features,label)):(id,label))

inputrddidfeatures=inputrdd.map(lambda (id,(features,label)):features)

inputrddNorm=inputrddidlabel.zip(normalizer1.transform(inputrddidfeatures))

print inputrddNorm.take(100)

inputrddNormData=inputrddNorm.filter(lambda (((Community,Year,Month),label),features):Year<2015 and label!="None" and label is not None)

inputrddtrain, inputrddtest =  inputrddNormData.randomSplit([0.7,0.3])

inputrddval=inputrddNorm.filter(lambda (((Community,Year,Month),label),features):Year>=2015)

DataTrain = inputrddtrain.map(parseDataF)
DataTest = inputrddtest.map(parseDataF)
DataVal = inputrddval.map(parseDataF)


model = RandomForest.trainClassifier(DataTrain, numClasses=5, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=20, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(DataTest.map(lambda x: x.features))
predictionsAndlabels = predictions.zip(DataTest.map(lambda lp: lp.label))


metrics = MulticlassMetrics(predictionsAndlabels)

labelsAndPredictions = DataTest.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(DataTest.count())
print('Test Error = ' + str(testErr))

print metrics.confusionMatrix().toArray()
print "precision = " + str(metrics.weightedPrecision)
print "recall = " + str(metrics.weightedRecall)


