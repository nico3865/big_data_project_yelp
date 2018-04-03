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
  data =ll.rdd.map(lambda x:(Vectors.dense(float(x[0]), float(x[1])),)).collect()
  df = spark.createDataFrame(data, ["features"])
  kmeans = KMeans(k=4, seed=1)
  model = kmeans.fit(df)

  # # Shows the result.
  centers = model.clusterCenters()
  print("Cluster Centers: ")
  for center in centers:
      print(center)

if __name__ == '__main__':
  main()