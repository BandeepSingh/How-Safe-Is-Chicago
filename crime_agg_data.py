from pyspark import SparkConf, SparkContext,SQLContext, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType,DateType
from pyspark.sql.functions import *
import sys, operator


conf = SparkConf().setAppName('How Safe is Chicago')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def convert(a):
	a[1].append(int(a[0]))
	return (a[1])
#input = "/Volumes/personal/uzmaa-irmacs/Chicago/data/all years Crimes_-_2001_to_present.csv"
#output="/users/uzmaa/Desktop/output"
input = sys.argv[1]
# input_social=sys.argv[2]
output = sys.argv[2]

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
    StructField('Year', StringType(), False),
    StructField('UpdatedOn', StringType(), False),
    StructField('Latitude', StringType(), False),
    StructField('Longitude', StringType(), False),
    StructField('Location', StringType(), False)
])

df = sqlContext.read.format('com.databricks.spark.csv').options(header='true').schema(schemaw).load(input)
# df=df.select(df["Community"],df["Year"],df["Arrest"])
df=df.select(df["LocationDesc"])
df.registerTempTable('crime_data')
result=sqlContext.sql("select Community,count(Community)as count from crime_data where Community!='' and Community!='0' and Year!='2016' group by Community,Year")
# result=sqlContext.sql("select Community,count(Arrest)as count from crime_data where Community!='' and Community!='0' and Year!='2016' and Arrest='true' group by Community,Year")

# Tag Cloud
#result=sqlContext.sql("select LocationDesc,count(LocationDesc)as count from crime_data group by LocationDesc")
#print result.rdd.map(list).collect()

crime_rdd=result.rdd.map(lambda l:(l[0],l[1])).groupByKey().map(lambda x : (x[0], list(x[1]))).map(convert).sortBy(lambda a:a[-1])
# print crime_rdd.collect()
crime_df=sqlContext.createDataFrame(crime_rdd)
crime_df.write.format('parquet').save(output)