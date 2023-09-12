// Import necessary libraries
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

// Create a Spark Session
val spark = SparkSession.builder().getOrCreate()

// Load Wholesale Customers Data from a CSV file
val dataset = spark.read.option("header", "true").option("inferSchema", "true").csv("Wholesale customers data.csv")

// Select specific columns for the training set: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
val feature_data = dataset.select("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")

// Create a VectorAssembler object to transform the features into a single 'features' column
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")

// Transform the feature_data using the assembler to create training_data
val training_data = assembler.transform(feature_data).select("features")

// Create a K-means Model with K=3
val kmeans = new KMeans().setK(3).setSeed(1L)

// Fit the K-means model to the training_data
val model = kmeans.fit(training_data)

// Display the cluster centers
println("Cluster Centers: ")
model.clusterCenters.foreach(println)