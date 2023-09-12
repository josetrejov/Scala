import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

// Set the error reporting level
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Creating a Spark Session
val spark = SparkSession.builder().getOrCreate()

// Reading the Titanic CSV file using Spark
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("titanic.csv")

// Printing the DataFrame schema
data.printSchema()

// Displaying example data
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example Data Row")
for(ind <- Range(1,colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}

// Preparing the DataFrame for Machine Learning
val logregdataall = data.select(data("Survived").as("label"), $"Pclass", $"Sex", $"Age", $"SibSp", $"Parch", $"Fare", $"Embarked")
val logregdata = logregdataall.na.drop()

// Handling Categorical Columns
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
val embarkIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkIndex")

val genderEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
val embarkEncoder = new OneHotEncoder().setInputCol("EmbarkIndex").setOutputCol("EmbarkVec")

// Assembling features
val assembler = (new VectorAssembler()
                  .setInputCols(Array("Pclass", "SexVec", "Age","SibSp","Parch","Fare","EmbarkVec"))
                  .setOutputCol("features") )

// Splitting the data into training and test sets
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)

// Setting up a pipeline for data processing and logistic regression
import org.apache.spark.ml.Pipeline

val lr = new LogisticRegression()

val pipeline = new Pipeline().setStages(Array(genderIndexer, embarkIndexer, genderEncoder, embarkEncoder, assembler, lr))

// Fitting the pipeline to the training data
val model = pipeline.fit(training)

// Applying the model to the test set
val results = model.transform(test)

// Model evaluation using MulticlassMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd

val metrics = new MulticlassMetrics(predictionAndLabels)

// Displaying the confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)