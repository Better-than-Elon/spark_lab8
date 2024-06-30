import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.functions.col
import db.{DbConnection}
import com.typesafe.scalalogging.Logger


object DataMart {
  private val logger = Logger("Logger")
  private val spark = SparkSession.builder
    .config("spark.app.name", "Clustering")
    .config("spark.driver.cores", 1)
    .config("spark.driver.maxResultSize", "1g")
    .config("spark.driver.memory", "2g")
    .config("spark.executor.memory", "2g")
    .config("spark.master", "local[*]")
	.config("spark.jars", "ojdbc8-21.5.0.0.jar")
	.config("spark.driver.extraClassPath", "ojdbc8-21.5.0.0.jar")
	.config("spark.executor.extraClassPath", "ojdbc8-21.5.0.0.jar")
    .getOrCreate()

  def readAndProccess(host: String): DataFrame = {
//    database.readTable("train_X")
//    val df = spark.read.option("header", "true").option("sep", "\t").option("inferSchema", "true").csv("truncated.csv").na.fill(0.0)
    val database = new DbConnection(spark, host)
    val df = database.readTable("train_X").na.fill(0.0)
    val inputCols: Array[String] = Array(
      "energy-kcal_100g",
      "sugars_100g",
      "energy_100g",
      "fat_100g",
      "saturated-fat_100g",
      "carbohydrates_100g"
    )

    val vec_assembler = new VectorAssembler()
      .setInputCols(inputCols).setOutputCol("features").setHandleInvalid("skip")

    //    val final_data = vec_assembler.transform(df_select)
    val final_data = vec_assembler.transform(df)
    
    logger.info("Data vectorized")

    val scaler = new StandardScaler().setInputCol("features")
      .setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)
    val scalerModel = scaler.fit(final_data)
    val scaled_final_data = scalerModel.transform(final_data)
    
    logger.info("Data scaled")
    scaled_final_data


  }
}