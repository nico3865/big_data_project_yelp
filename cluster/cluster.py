import os
import sys
from numpy import array
from math import sqrt
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from  pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
def main():
  spark = SparkSession.Builder().getOrCreate()
  datapath = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
  dataset = spark.read.format('libsvm').json(datapath+'/data/business.json')
  ll = dataset.select(dataset.longitude, dataset.latitude)
  trainingData=ll.rdd.map(lambda x:(Vectors.dense(x[0], x[1]), 0)).toDF(["features", "label"])
  print(trainingData.show())
  # dataset = rdd.map(lambda row: (Vectors.dense([float(row['longitude']), float(row['latitude'])])))
  
  # create dataframe
  # df = spark.createDataFrame(dataset, ['Features'])
  # Build the model (cluster the data)
  kmeans = KMeans(featuresCol='features').setK(2).setSeed(1)
  model = kmeans.fit(trainingData)

  # Make predictions
  predictions = model.transform(dataset)

  # Evaluate clustering by computing Silhouette score
  evaluator = ClusteringEvaluator()

  silhouette = evaluator.evaluate(predictions)
  print("Silhouette with squared euclidean distance = " + str(silhouette))

  # Shows the result.
  centers = model.clusterCenters()
  print("Cluster Centers: ")
  for center in centers:
      print(center)

if __name__ == '__main__':
  main()