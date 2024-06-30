import os
import findspark
from spark_builder import SparkBuilder
from data_processing import DataProcessing
from pyspark.sql.functions import explode, split, col
import yaml
import sys
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from logger import Logger

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

cfg = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
logger = Logger(show=True)
log = logger.get_logger(__name__)

if __name__ == '__main__':
    spark = SparkBuilder({'spark.app.name': 'Kmenas'}).getSession()

    processing = DataProcessing(spark)

    dataset = processing.load()

    table_struct = ', '.join([f'`{col}` FLOAT'for col in dataset.columns])
    url = f"jdbc:oracle:thin:@{cfg['db_connection']['host']}:1521/FREE"
    properties = {
        "user": "system",
        "password": str(cfg['db_connection']['pass']),
        "driver": "oracle.jdbc.driver.OracleDriver"
    }

    dataset.write.option("createTableColumnTypes", table_struct).jdbc(
        url=url, table=cfg['db']['data_table'], mode="overwrite", properties=properties)
    log.info(f'Dataset uploaded to db: {table_struct}')
    del(dataset)
    
    log.info("Creating session...")
    proc_data = spark._jvm.DataMart.readAndProccess(cfg['db_connection']['host'])
    log.info("Processing data...")
    final_data = DataFrame(proc_data, spark).select("scaledFeatures")
    log.info(f'Data from DataMart\n{final_data.select("scaledFeatures")._jdf.showString(5, 40, False)}')
    evaluator = ClusteringEvaluator(predictionCol='prediction',
                                featuresCol='scaledFeatures',
                                metricName='silhouette',
                                distanceMeasure='squaredEuclidean')
    kmeans = KMeans(featuresCol='scaledFeatures', k=12, seed=42)
    model = kmeans.fit(final_data)
    predictions = model.transform(final_data)
    score = evaluator.evaluate(predictions)
    log.info(f'Predictions\n{predictions.select("prediction")._jdf.showString(5, 40, False)}')
    log.info(f"Kmeans score: {score}")
    
    url = f"jdbc:oracle:thin:@{cfg['db_connection']['host']}:1521/FREE"
    properties = {
        "user": "system",
        "password": str(cfg['db_connection']['pass']),
        "driver": "oracle.jdbc.driver.OracleDriver"
    }
    
    predictions.select(['prediction']).write.jdbc(url=url, table=cfg['db']['pred_table'], mode='append', properties=properties)
    log.info(f"Predictions saved")
    
    spark.stop