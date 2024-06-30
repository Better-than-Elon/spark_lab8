package db

import org.apache.spark.sql.{DataFrame, SparkSession}


class DbConnection(spark: SparkSession, host: String) {
  private val JDBC_URL = "jdbc:oracle:thin:@" + host + ":1521/FREE"
  private val USER = "system"
  private val PASSWORD = "12345"

  def readTable(tablename: String): DataFrame = {
    spark.read
      .format("jdbc")
      .option("url", JDBC_URL)
      .option("user", USER)
      .option("password", PASSWORD)
      .option("dbtable", tablename)
      .option("inferSchema", "true")
	  .option("driver", "oracle.jdbc.driver.OracleDriver")
      .load()
  }

  def insertDf(df: DataFrame, tablename: String): Unit = {
    df.write
      .format("jdbc")
	  .option("driver", "oracle.jdbc.driver.OracleDriver")
      .option("url", JDBC_URL)
      .option("user", USER)
      .option("password", PASSWORD)
      .option("dbtable", tablename)
      .mode("append")
      .save()
  }
}