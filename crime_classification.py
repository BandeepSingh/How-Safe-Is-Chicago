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

conf = SparkConf().setAppName('How Safe is Chicago')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

#input = "/Volumes/personal/uzmaa-irmacs/Chicago/data/all years Crimes_-_2001_to_present.csv"
#output="/users/uzmaa/Desktop/output"
input = sys.argv[1]
output = sys.argv[2]
# input_social=sys.argv[2]
# output = sys.argv[3]

schemaw = StructType([
    StructField('ID', StringType(), False),
    StructField('CaseNumber', StringType(), False),
    StructField('Date', StringType(), False),
    StructField('Block', StringType(), False),
	StructField('IUCR', StringType(), False),
    StructField('PrimaryType', StringType(), False),
    StructField('Description', StringType(), False),
    StructField('LocationDesc', StringType(), False),
    StructField('Arrest', StringType(), False),
    StructField('Domestic', StringType(), False),
    StructField('Beat', StringType(), False),
    StructField('District', StringType(), False),
    StructField('Ward', StringType(), False),
    StructField('Community', StringType(), False),
    StructField('FBICode', StringType(), False),
    StructField('XCoordinate', StringType(), False),
    StructField('YCoordinate', StringType(), False),
    StructField('Year', IntegerType(), False),
    StructField('UpdatedOn', StringType(), False),
    StructField('Latitude', StringType(), False),
    StructField('Longitude', StringType(), False),
    StructField('Location', StringType(), False)
])

df = sqlContext.read.format('com.databricks.spark.csv').options(header='true').schema(schemaw).load(input)

distinct_crime=df.select(df.PrimaryType).distinct().map(tuple).map(lambda l:l[0]).collect()
# print distinct_crime

crime_types={}
count=1
for crime in distinct_crime:
    crime_types[crime]=count
    count+=1



def transform(data):
    transformed_data=crime_types[data]
    return transformed_data
	
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
	return LabeledPoint(line[0],line[1:10])
df.registerTempTable('data_source')
#df_selected=df.select(df.PrimaryType,df.IUCR,df.Arrest,df.Domestic,df.Beat,df.District,df.Ward,df.Community,df.FBICode)
df_selected=sqlContext.sql("""select ds.PrimaryType,IF( ds.Arrest='false',0,1) as bool_arrest,IF( ds.Domestic='false',0,1) as bool_domestic,ds.Beat,ds.District,ds.Ward,ds.Community,ds.FBICode from data_source ds where ds.Arrest!=' ' and ds.Domestic!=' ' and ds.Beat!=' ' and ds.District!=' ' and ds.Ward!=' ' and ds.Community!=' ' and ds.FBICode!=' ' and ds.IUCR!='08B' and ds.Beat!='08B' and ds.District!='08B' and ds.Ward!='08B' and ds.Community!='08B' and ds.FBICode!='08B' and ds.IUCR!='08A' and ds.Beat!='08A' and ds.District!='08A' and ds.Ward!='08A' and ds.Community!='08A' and ds.FBICode!='08A'""");
#df_selected=sqlContext.sql("""select distinct(ds.IUCR) from data_source ds""");

slen=udf(transform, IntegerType())
df1=df_selected.withColumn("label", slen(df_selected.PrimaryType))
#df1.show()
df1.write.format('parquet').save(output)