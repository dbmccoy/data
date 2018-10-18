// Databricks notebook source
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
val df = spark.sql("select * from netflix")


// COMMAND ----------

// What are the column names?

df.columns

// What does the Schema look like?

df.printSchema()

// Print out the first 5 columns.

df.head(5)

// Use describe() to learn about the DataFrame.

df.describe().show

// COMMAND ----------

// Create a new dataframe with a column called HV Ratio that
// is the ratio of the High Price versus volume of stock traded
// for a day.

val df2 = df.withColumn("HV Ratio", $"High" / $"Close").show

// COMMAND ----------

// What day had the Peak High in Price?

df.orderBy($"High".desc).show(1)

// What is the mean of the Close column?

df.select(mean("Close")).show

// What is the max and min of the Volume column?

df.select(max("Volume")).show
df.select(min("Volume")).show

// COMMAND ----------

// How many days was the Close lower than $ 600?

df.filter("Close < 600").count()

// What percentage of the time was the High greater than $500 ?

(df.filter("High > 500").count()*1.0 / df.count())*100

// What is the Pearson correlation between High and Volume?

df.select(corr("High","Volume")).show

// What is the max High per year?

val df2 = df.withColumn("Year",year($"Date"))
val df3 = df2.select($"Year",$"High").groupBy("Year").max() 
df3.select($"Year",$"max(High)").show()
// What is the average Close for each Calender Month?

val df4 = df.withColumn("Month",month($"Date"))
val df5 = df4.select($"Month",$"Close").groupBy("Month").mean()
df5.select($"Month",$"avg(Close)").show()
