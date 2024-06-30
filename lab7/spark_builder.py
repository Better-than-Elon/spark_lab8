from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import yaml


class SparkBuilder:
    def __init__(self, update_cfg=None, config_path='config.yaml'):
        with open(config_path) as f:
            spark_cfg = yaml.load(f, Loader=yaml.FullLoader)['spark']
            if update_cfg is not None:
                spark_cfg.update(update_cfg)
            conf = SparkConf().setAll(spark_cfg.items())
            self.spark = SparkSession.builder.config(conf=conf).getOrCreate()

    def getSession(self):
        return self.spark

    def stop(self):
        self.spark.stop()
