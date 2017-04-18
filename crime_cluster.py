from pyspark import SparkConf, SparkContext,SQLContext, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType,DateType
from pyspark.sql.functions import *
from pyspark.mllib.clustering import KMeans, KMeansModel
import sys
from math import sqrt


conf = SparkConf().setAppName('How Safe is Chicago')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

inputs = sys.argv[1]
data_read = sqlContext.read.parquet(inputs)
#data_read.show()

def sort_data(data):
    data.sort()
    return data

def main():
    # features=data_read#.rdd.map(lambda row: row[0]).cache()
    data_read.registerTempTable('data')#.rdd.map(tuple).map(lambda (a,b):(int(a),b)).sortBy(lambda (a,b):a)
    avg_data=sqlContext.sql("""
     SELECT avg(_1) AS 2001_avg,avg(_2) AS 2002_avg,avg(_3) AS 2003_avg,avg(_4) AS 2004_avg,avg(_5) AS 2005_avg,avg(_6) AS 2006_avg,avg(_7) AS 2007_avg,avg(_8) AS 2008_avg,avg(_9) AS 2009_avg,avg(_10) AS 2010_avg, avg(_11) AS 2011_avg,avg(_12) AS 2012_avg,avg(_13) AS 2013_avg,avg(_14) AS 2014_avg,avg(_15) AS 2015_avg
     FROM data
     """)
    # avg_data.show()
    avg_data.registerTempTable('avg_data')
    mod_data=sqlContext.sql("""
     SELECT IF(dt._1>av.2001_avg,1,0) AS yr_2001,IF(dt._2>av.2002_avg,1,0) AS yr_2002,IF(dt._3>av.2003_avg,1,0) AS yr_2003,IF(dt._4>av.2004_avg,1,0) AS yr_2004,IF(dt._5>av.2005_avg,1,0) AS yr_2005,IF(dt._6>av.2006_avg,1,0) AS yr_2006,IF(dt._7>av.2007_avg,1,0) AS yr_2007,IF(dt._8>av.2008_avg,1,0) AS yr_2008,IF(dt._9>av.2009_avg,1,0) AS yr_2009,IF( dt._10>av.2010_avg,1,0) AS yr_2010,IF( dt._11>av.2011_avg,1,0) AS yr_2011,IF( dt._12>av.2012_avg,1,0) AS yr_2012,IF( dt._13>av.2013_avg,1,0) AS yr_2013,IF( dt._14>av.2014_avg,1,0) AS yr_2014,IF( dt._15>av.2015_avg,1,0) AS yr_2015
     FROM data dt,avg_data av order by dt._16
     """)
    parsedData=mod_data.rdd.map(list).map(sort_data)

    # print mod_rdd.collect()

    # Evaluate clustering by computing Within Set Sum of Squared Errors
    k=3
    model = KMeans.train(parsedData, k, maxIterations=100, runs=10, initializationMode="random")
    print model.predict(parsedData).collect()
    # def error(point):
    #     center = model.centers[model.predict(point)]
    #     return center
    #     # return(point - center)
    #     # return sqrt(sum([x**2 for x in (point - center)]))
    #
    # WSSSE = parsedData.map(lambda point: error(point))#.reduce(lambda x, y: x + y)
    # print WSSSE.collect()
    # # print("Within Set Sum of Squared Error = " + str(WSSSE))

if __name__ == "__main__":
    main()