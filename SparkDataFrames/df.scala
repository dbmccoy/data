import org.apache.spark.sql.SparkSession

val c = "c"
val spark = SparkSession.builder().getOrCreate()

val df = spark.read.csv("CitiGroup2006_2008")

//df.head(5)
