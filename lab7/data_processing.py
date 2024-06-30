import yaml
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from logger import Logger

class DataProcessing:
    def __init__(self, spark):
        self.spark = spark
        logger = Logger(show=True)
        self.log = logger.get_logger(__name__)

    def load(self, config_path='config.yaml'):
        cfg = yaml.load(open(config_path), Loader=yaml.FullLoader)
        dataset = self.spark.read.csv(cfg['dataset']['data_path'], header=True, inferSchema=True, sep='\t')
        dataset = dataset.select(cfg['dataset']['filtered_cols'])
        dataset = dataset.na.fill(value=0)
        
        self.log.info(f'Input data\n{dataset._jdf.schema().treeString()}')
        return dataset

    def vectorize(self, dataset=None, outputCol = 'features'):
        if dataset is None:
            dataset = self.load()
        vec_assembler = VectorAssembler(inputCols=dataset.columns,
                                outputCol=outputCol)
        vec_dataset = vec_assembler.transform(dataset)
        
        self.log.info(f'VectorAssembler\n{vec_dataset.select(outputCol)._jdf.showString(5, 40, False)}')
        return vec_dataset
    
    def scale(self, vec_dataset=None, inputCol='features', outputCol='scaledFeatures'):
        if vec_dataset is None:
            vec_dataset = self.vectorize()
        scaler = StandardScaler(inputCol=inputCol,
                        outputCol=outputCol,
                        withStd=True,
                        withMean=False)
        scalerModel = scaler.fit(vec_dataset)
        scaled_dataset = scalerModel.transform(vec_dataset)
        
        self.log.info(f'StandardScaler\n{scaled_dataset.select(outputCol)._jdf.showString(5, 40, False)}')
        return scaled_dataset
    
    def cluster(self, scaled_dataset=None, predictionCol='prediction', featuresCol='scaledFeatures', k=12):
        if scaled_dataset is None:
            scaled_dataset = self.scale()
        
        evaluator = ClusteringEvaluator(predictionCol=predictionCol,
                                featuresCol=featuresCol,
                                metricName='silhouette',
                                distanceMeasure='squaredEuclidean')

        kmeans = KMeans(featuresCol=featuresCol, k=k, seed=42)
        model = kmeans.fit(scaled_dataset)
        predictions = model.transform(scaled_dataset)
        score = evaluator.evaluate(predictions)
        
        self.log.info(f'Silhouette Score for k = {k} is {score}')
        return predictions, score