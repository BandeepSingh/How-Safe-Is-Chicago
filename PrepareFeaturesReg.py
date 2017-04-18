__author__ = 'uzmaa'
from pyspark import SparkConf, SparkContext,SQLContext, Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType,DateType
from pyspark.sql.functions import *
import sys, operator
from pyspark.sql.window import Window
from pyspark.mllib.linalg import Vectors

conf = SparkConf().setAppName('How Safe is Chicago')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

input = sys.argv[1] + "/all years Crimes_-_2001_to_present.csv"
input_ct = sys.argv[1] + "/CrimeTypes.csv"
output=sys.argv[2]
#sample crime data.txt"
#input = sys.argv[1]
#input_social=sys.argv[2]
#output = sys.argv[3]

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
schemact = StructType([
    StructField('ctID', IntegerType(), False),
    StructField('ctName', StringType(), False),
])
df = sqlContext.read.format('com.databricks.spark.csv').options(header='false').schema(schemaw).load(input).cache()
df_ct = sqlContext.read.format('com.databricks.spark.csv').options(header='false').schema(schemact).load(input_ct)

df_f=df.select(df["Community"],df["Year"],substring(df["Date"],1,2).alias("Month"),df["PrimaryType"])
df_f.registerTempTable('crime_data')
df_ct.registerTempTable('crime_types')
df_ct.show()

result=sqlContext.sql("select Community,Year,Month,c.ctID,count(*) as count from crime_data d inner join crime_types c on d.PrimaryType=c.ctName  where Community!='' group by Community,Year,Month,c.ctID order by Year,Month,Community,c.ctID")

result.show()

def convertToSparse(a):

    itemNo=[]
    item=[]
    j=0

    for i in a:
       j=j+1
       if j%2==0:
           item.append(int(i))
       else:
           itemNo.append(int(i))

    return Vectors.sparse(36,itemNo,item)

crime_rdd=result.rdd.map(lambda l:((int(l[0]),int(l[1]),int(l[2])),(l[3],l[4]))).reduceByKey(lambda a,b:a+b).mapValues(convertToSparse).coalesce(1)

df_l=df.select(df["Community"],to_date(concat_ws("-",df["Year"],substring(df["Date"],1,2),lit("01"))).alias("DateThis"))

df_l=df_l.select(df_l["Community"],df_l["DateThis"],add_months(df_l["DateThis"],-12).alias("LastYearDate"))

df_l.registerTempTable('labelData')

result=sqlContext.sql("select Community,Year(LastYearDate) as LastYear,Month(LastYearDate) as LastYearMonth,count(*) as NextYearCount from labelData  where Community!='' group by Community,Year(LastYearDate),Month(LastYearDate)")

crimelbl_rdd=result.rdd.map(lambda l:((int(l[0]),int(l[1]),int(l[2])),l[3]))

#print crimelbl_rdd.take(5)

rddJoined=crime_rdd.leftOuterJoin(crimelbl_rdd).coalesce(1)

rddJoined.saveAsPickleFile(output)


