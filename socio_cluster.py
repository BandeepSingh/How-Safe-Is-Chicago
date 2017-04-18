from pyspark import SparkConf, SparkContext,SQLContext, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType,DateType
from pyspark.sql.functions import *
import sys, operator
from pyspark.mllib.feature import Normalizer
from pyspark.mllib.clustering import KMeans, KMeansModel

conf = SparkConf().setAppName('How Safe is Chicago')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def convert_type(a):
	b=map(float, a[1][1])
	c=a[1][0]+b
	d=[]
	d.append(int(a[0]))
	d=d+c
	return (d)

social_inputs = sys.argv[1]
health_input=sys.argv[2]
rddSocial = sc.textFile(social_inputs).map(lambda l:(l.split(',')[0],l.split(',')[2:]))
#print rddSocial.collect()
rddHealth = sc.textFile(health_input).map(lambda l:(l.split(',')[0],l.split(',')[2:4]))
#print rddHealth.collect()
rddJoined=rddSocial.join(rddHealth)
rddJoined=rddJoined.map(convert_type).sortBy(lambda l:l[0])

schemaString = "area_code housing_crowd below_poverty under_employ degreeless under_aged per_capita_income hardship_index birth_rate fertility_rate"
fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
schema = StructType(fields)

new_df=sqlContext.createDataFrame(rddJoined,schema)

new_df.registerTempTable("social_health_index")
avg_df = sqlContext.sql("SELECT avg(housing_crowd) as hosuing_avg,avg(below_poverty) as below_poverty_avg ,avg(under_employ) as under_employ_avg,avg(degreeless) as degree_avg,avg(under_aged) as under_aged_avg,avg(per_capita_income) as per_capita_avg,avg(hardship_index) as hardship_index_avg,avg(birth_rate) as birth_rate_avg,avg(fertility_rate) as fertility_rate_avg FROM social_health_index")
avg_df.registerTempTable("avg_data")

# result=sqlContext.sql("SELECT IF(ad.hosuing_avg>sh.housing_crowd, 0, 1) as housing_bool,IF(ad.below_poverty_avg>sh.below_poverty, 0, 1) as below_poverty_bool,IF(ad.under_employ_avg>sh.under_employ, 0, 1) as under_employ_bool,IF(ad.degree_avg>sh.degreeless, 0, 1) as degree_bool,IF(ad.under_aged_avg>sh.under_aged, 0, 1) as under_aged_bool,IF(ad.per_capita_avg>sh.per_capita_income, 0, 1) as per_capita_bool,IF(ad.hardship_index_avg>sh.hardship_index, 0, 1) as hardship_index_bool,IF(ad.birth_rate_avg>sh.birth_rate, 0, 1) as birth_rate_bool,IF(ad.fertility_rate_avg>sh.fertility_rate, 0, 1) as fertility_rate_bool  FROM social_health_index sh,avg_data ad order by sh.area_code")
# features= result.rdd.map(tuple).cache()
#
# model = KMeans.train(features, 3, maxIterations=40, runs=10, initializationMode="random", seed=20)
# print model.predict(features).collect()
