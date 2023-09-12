// Import necessary libraries for Spark and Logistic Regression
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Create a Spark Session
val spark = SparkSession.builder().getOrCreate()

// Use Spark to read in the Advertising csv file.
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("advertising.csv")

// Print the Schema of the DataFrame
//data.printSchema()

// Print out a sample row of the data (multiple ways to do this)
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example Data Row")
for(ind <- Range(1,colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}

// Renaming the "Clicked on Ad" column to "label"
// Selecting specific columns for analysis
val timedata = data.withColumn("Hour", hour(data("Timestamp")))
val logregdataall = (timedata.select(data("Clicked on Ad").as("label"),
                    $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male"))
val logregdata = logregdataall.na.drop()

// Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

// Creating a VectorAssembler to assemble feature columns into a "features" column
val assembler = (new VectorAssembler()
                  .setInputCols(Array("Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Hour"))
                  .setOutputCol("features") )

// Splitting the data into training and test sets
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)

// Import Pipeline
import org.apache.spark.ml.Pipeline

// Creating a LogisticRegression object
val lr = new LogisticRegression()

// Creating a pipeline with stages: assembler and logistic regression
val pipeline = new Pipeline().setStages(Array(assembler, lr))

// Fitting the pipeline to the training set
val model = pipeline.fit(training)

// Transforming the test set to get results
val results = model.transform(test)

// Importing MulticlassMetrics for metrics and evaluation
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Converting test results to an RDD
val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd

// Creating a MulticlassMetrics object
val metrics = new MulticlassMetrics(predictionAndLabels)

// Printing the Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)