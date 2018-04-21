# -*- coding: utf-8 -*-

import numpy as np

from pyspark.ml.linalg import Vectors

from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


#------ als:

import sys
import os
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer

from constants import SEED

def get_user_business(rating, user_mean, item_mean, rating_global_mean):
    return rating-(user_mean +item_mean-rating_global_mean)

def get_final_ratings(i, user_mean, item_mean, global_average_rating):
    final_ratings = i+user_mean+item_mean-global_average_rating
    return final_ratings


spark = SparkSession.Builder().getOrCreate()
seed = 1  # int(sys.argv[SEED])
# datapath = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
# rdd = spark.read.json(datapath+'/data/review_truncated_RAW.json').rdd

# filename = '/Users/nicolasg-chausseau/Downloads/yelp_dataset/review.json'
# filename = '../data/review_50K_0.json'
filename = 'review_50K_0.json'
# filename = '/Users/nicolasg-chausseau/Downloads/yelp_dataset/review_MTL_ONLY.json'
# filename = '/Users/nicolasg-chausseau/big_data_project_yelp/data/review_truncated_RAW.json'
rdd = spark.read.json(filename).limit(1200).rdd # datapath+'/data/review_trunca®ted_RAW.json'
# TODO: put the limit above back to 100,000

df = spark.createDataFrame(rdd)
# df.show()
# sys.exit()
(training, test) = df.randomSplit([0.8, 0.2], seed) #df.randomSplit([0.8, 0.2], seed)
userIdRdd1 = test.select('user_id').rdd.distinct().zipWithIndex().map(lambda x: (x[0][0], x[1]))
businessIdRdd1 = test.select('business_id').rdd.distinct().zipWithIndex().map(lambda x: (x[0][0], x[1]))

# convert to dataframe
userIdDf2 = spark.createDataFrame(userIdRdd1) \
    .withColumnRenamed('_1', 'user_id') \
    .withColumnRenamed('_2', 'user_id_indexed')
businessIdDf2 = spark.createDataFrame(businessIdRdd1) \
    .withColumnRenamed('_1', 'business_id') \
    .withColumnRenamed('_2', 'business_id_indexed')

# join user id zipped with index and business id with index
test = test.join(userIdDf2, ['user_id'], 'left').join(businessIdDf2, ['business_id'], 'left')

# get user mean
user_mean = training.groupBy('user_id').mean('stars').withColumnRenamed('avg(stars)', 'user-mean')

# get item mean
business_mean = training.groupBy('business_id').mean('stars').withColumnRenamed('avg(stars)', 'business-mean')

# join user mean df and training df
training = training.join(user_mean, ['user_id']) \
    .select(training['user_id'], training['business_id'], training['stars'], user_mean['user-mean'])

# join item mean df and traning df
training = training.join(business_mean, ['business_id']) \
    .select(training['user_id'], training['business_id'], training['stars'],
            user_mean['user-mean'], business_mean['business-mean'])

# get global average
rating_global_average = training.groupBy().avg('stars').head()[0]

# add user item interaction to training column
training = training.withColumn('user-business-interaction',
                               get_user_business(training['stars'],
                                                 user_mean['user-mean'],
                                                 business_mean['business-mean'],
                                                 rating_global_average))

# convert distinct user ids and business ids to integer
userIdRdd = training.select('user_id').rdd.distinct().zipWithIndex().map(lambda x: (x[0][0], x[1]))
businessIdRdd = training.select('business_id').rdd.distinct().zipWithIndex().map(lambda x: (x[0][0], x[1]))

# convert to dataframe
userIdDf = spark.createDataFrame(userIdRdd) \
    .withColumnRenamed('_1', 'user_id') \
    .withColumnRenamed('_2', 'user_id_indexed')
businessIdDf = spark.createDataFrame(businessIdRdd) \
    .withColumnRenamed('_1', 'business_id') \
    .withColumnRenamed('_2', 'business_id_indexed')
# join user id zipped with index and business id with index
training = training.join(userIdDf, ['user_id'], 'left').join(businessIdDf, ['business_id'], 'left')
als = ALS(maxIter=5,
          rank=70,  # ORIGINAL
          # rank=3,
          regParam=0.01,
          # regParam=0.1,
          userCol='user_id_indexed',
          itemCol='business_id_indexed',
          ratingCol='user-business-interaction',
          coldStartStrategy='drop')
als.setSeed(seed)
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
# test.show() # I should make my cross product have the same columns
# predictions.show()
predictions = predictions.join(user_mean, ['user_id'],'left')
predictions = predictions.join(business_mean, ['business_id'], 'left')
rating_global_mean = training.groupBy().mean('stars').head()[0]
predictions = predictions.na.fill(rating_global_mean)
final_stars = predictions.withColumn('final-stars', get_final_ratings(predictions['prediction'],
                                                                      predictions['user-mean'],
                                                                      predictions['business-mean'],
                                                                      rating_global_mean))
evaluator = RegressionEvaluator(metricName='rmse',
                                labelCol='stars',
                                predictionCol='final-stars')
rmse = evaluator.evaluate(final_stars)
print(float(rmse))







#------ /als


# --> this seems good too:, instead of model.transofmr(testset)
# model.predictAll(testset).collect()



# now PCA: https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/

spark = SparkSession.Builder().getOrCreate()


# #read the dataset and plot a scatter graph between 1st and 2nd variable
# import matplotlib.pyplot as plt
# iris = datasets.load_iris()
# data = iris.data
# target = iris.target
# setosa = data[target==0]
# versicolor = data[target==1]
# verginica = data[target==2]
# plt.scatter(setosa[:,0], setosa[:,1], c="b",label="setosa")
# plt.scatter(versicolor[:,0], versicolor[:,1], c="g",label="versicolor")
# plt.scatter(verginica[:,0], verginica[:,1], c="r",label="verginica")


# necesary imports

from pyspark.mllib.linalg.distributed import IndexedRowMatrix, IndexedRow
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as f

# numpy array -> rdd -> dataframe


# rdd = spark.sparkContext.parallelize(iris_data.tolist()).zipWithIndex()
# iris_df =
# spark.createDataFrame(rdd).toDF("features","id")
# n = rdd.count()
# p = len(rdd.take(1)[0][0])

# change the data type of features to vectorUDT from array[double]

# udf_change = f.udf(lambda x: Vectors.dense(x), VectorUDT())
#
# iris_df = iris_df.withColumn("features", udf_change("features"))



# # create the standard scaler model
# stdScaler = StandardScaler(withMean = True, withStd = True, inputCol="features", outputCol="scaled_features")
# #fit the model on the dataset
# model = stdScaler.fit(iris_df)
# # transform the dataset
# iris_std_df = model.transform(iris_df).drop("features").withColumnRenamed("scaled_features","features")

# all_business_user_pairs = df.select("business_id","user_id").rdd.map(lambda x: (x[0], x[1]))
# predictions = model.predictAll(all_business_user_pairs)
# turns out this is the RDD API, called ml, not mllib. great. stupid rather.


print("#####################")
final_stars_FINAL = final_stars.select("business_id","user_id","final-stars")
final_stars_FINAL.show()
print("#####################")
# sys.exit()

# first must get the cross product of all users by all businesses:
list_of_user_ids = final_stars_FINAL.rdd.map(lambda p: p[1].strip())
list_of_user_ids_distinct = list_of_user_ids.distinct()
list_of_user_ids_distinct_MAP_COLLECTED = list_of_user_ids_distinct.map(lambda x: (x, 0.0)).collect()
# print("do I have a good list of distinct user ids?")
# for item in list_of_user_ids_distinct_MAP_COLLECTED:
#     print(item)
# print("do I have a good list of distinct user ids?")

list_of_business_ids = final_stars_FINAL.rdd.map(lambda p: p[0].strip())
list_of_business_ids_distinct = list_of_business_ids.distinct()
list_of_business_ids_distinct_MAP_COLLECTED = list_of_business_ids_distinct.map(lambda x: (x, 0.0)).collect()

# get cross product:
cartesian_product = list_of_business_ids.cartesian(list_of_user_ids_distinct)
# print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
# for item in cartesian_product.collect():
#     print(item)
# print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
# sys.exit()
cartesian_product_DF = cartesian_product.toDF(["business_id","user_id"])#.withColumn()

# add all the necessary columns for the ALS model to do its job:
userIdRdd1_2 = cartesian_product_DF.select('user_id').rdd.distinct().zipWithIndex().map(lambda x: (x[0][0], x[1]))
businessIdRdd1_2 = cartesian_product_DF.select('business_id').rdd.distinct().zipWithIndex().map(lambda x: (x[0][0], x[1]))

# convert to dataframe
userIdDf2_2 = spark.createDataFrame(userIdRdd1_2) \
    .withColumnRenamed('_1', 'user_id') \
    .withColumnRenamed('_2', 'user_id_indexed')
businessIdDf2_2 = spark.createDataFrame(businessIdRdd1_2) \
    .withColumnRenamed('_1', 'business_id') \
    .withColumnRenamed('_2', 'business_id_indexed')

# join user id zipped with index and business id with index
cartesian_product_DF = cartesian_product_DF.join(userIdDf2_2, ['user_id'], 'left').join(businessIdDf2_2, ['business_id'], 'left')
cartesian_product_DF.show()


# .... this below (prepare) is not enough
# of course --> do like for LA3:
#     make a separate rdd with the list of distinct users .. for EVERY business!
#     then add it as extra rows to the date_add
#     them merge by key, and write my own function to merge the lists correctly, respecting the order
#     ... this will take forever to complete, should run on orwell ..... we'll see.'


# data preparation: get the searchable map of .... state --> plants # TODO: keep a searchable table for business and user ids which will become ints ...
searchable_plant_in_state_sets_1 = cartesian_product_DF.rdd.map(lambda x: (x[0], [(x[1], float(x[2]))]))
searchable_plant_in_state_sets_2 = searchable_plant_in_state_sets_1.reduceByKey(lambda a, b: a + b)
def mergeListsOfTuples(a, b_reference):
    from collections import defaultdict
    a_dict = defaultdict(lambda:0.0, a)
    b_dict_reference = defaultdict(lambda:0.0, b_reference)
    for key, value in b_reference:
        b_dict_reference[key] += float(a_dict[key])
    list_to_return = [(k,v) for k,v in b_dict_reference.items()]
    return list_to_return
sqlContext = SQLContext(spark.sparkContext)
sqlContext.udf.register("mergeListsOfTuples", mergeListsOfTuples)
searchable_plant_in_state_sets_3_pre = searchable_plant_in_state_sets_2.map(lambda x: (x[0], mergeListsOfTuples(x[1], list_of_user_ids_distinct_MAP_COLLECTED)))
# print("do I have a good list of searchable_plant_in_state_sets_3_pre?")
# for item in searchable_plant_in_state_sets_3_pre.collect():
#     print(item)
# print("do I have a good list of searchable_plant_in_state_sets_3_pre?")
# searchable_plant_in_state_sets_3 = searchable_plant_in_state_sets_3_pre.filter(lambda x: x[0] in all_states)
searchable_plant_in_state_sets_3 = searchable_plant_in_state_sets_3_pre.map(lambda x: (x[0], sorted(x[1], key=lambda x: x[0], reverse=True)))
searchable_plant_in_state_sets_4 = searchable_plant_in_state_sets_3.map(lambda x: (x[0], [p[1] for p in x[1]]))
print("do I have a good list of searchable_plant_in_state_sets_4?")
for item in searchable_plant_in_state_sets_3_pre.collect():
    print(item)
print("do I have a good list of searchable_plant_in_state_sets_4?")
final_stars_FINAL_READY = searchable_plant_in_state_sets_4.map(lambda x: x[1]).zipWithIndex() # get rid of business_ids and replace with simple integer ids instead.
# final_stars_FINAL_READY = searchable_plant_in_state_sets_5.map(lambda x: x[1])

# # prepare:
# # iris_irm = IndexedRowMatrix(iris_std_df.rdd.map(lambda x: IndexedRow(x[0], x[1].tolist())))
# # rdd.map(lambda (k, v): (k, sorted(v, key=lambda x: x[1], reverse=True)))
# final_stars_FINAL_READY = final_stars_FINAL.rdd\
#     --.map(lambda x: (x[0], [(x[1], x[2])]))\
#     --.reduceByKey(lambda a,b: a+b)\
#     --.map(lambda x: (x[0], sorted(x[1], key=lambda x: x[0], reverse=True)))\
#     --.map(lambda x: (x[0], [p[1] for p in x[1]]))\
#     --.map(lambda x: x[1])\
#     --.zipWithIndex()


# training = training.drop("user-mean")
# training = training.drop("business-mean")
# data needs to have following fields:
# |         business_id|             user_id|cool|      date|funny|           review_id|stars|                text|useful|user_id_indexed|business_id_indexed|
# predictions = model.transform(training) # predicting only on training data ... with all zeros ... I don't know how relevant it is ... we'll have to compare its explainedVariance with the full predicted matrix.
predictions = model.transform(final_stars_FINAL_READY) # df... but it would have to be prepared too. for now I can achieve the same by making the test set so small that training is almost all the ratings.
# do the same again: adjust for biases:
predictions = predictions.join(user_mean, ['user_id'],'left')
predictions = predictions.join(business_mean, ['business_id'], 'left')
rating_global_mean = training.groupBy().mean('stars').head()[0]
predictions.show()
predictions = predictions.na.fill(rating_global_mean)
final_stars_FINAL = predictions.withColumn('final-stars', get_final_ratings(predictions['prediction'],
                                                                            predictions['user-mean'],
                                                                            predictions['business-mean'],
                                                                            rating_global_mean))



# do I have a 2D matrix now?
print("# do I have a 2D matrix now --> FULLY PREDICTED ????????????????????????")
for item in final_stars_FINAL_READY.collect():
    print(item)
print("# do I have a 2D matrix now --> FULLY PREDICTED ??????????????????????? ==> NOW WE KNOw .........")
iris_irm = IndexedRowMatrix(final_stars_FINAL_READY.map(lambda x: IndexedRow(x[1], x[0])))


# https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/
# do SVD:
num_of_top_sing_values = 2
SVD = iris_irm.computeSVD(num_of_top_sing_values, True)

U = SVD.U
S = SVD.s.toArray()

# compute the eigenvalues and number of components to retain
n = final_stars_FINAL_READY.count()
eigvals = S**2/(n-1)
eigvals = np.flipud(np.sort(eigvals))
cumsum = eigvals.cumsum()
total_variance_explained = cumsum/eigvals.sum()
print("total_variance_explained =======================================> ", total_variance_explained)
# on 1000 with 2 PCs --> total_variance_explained =======================================>  [0.61812207 1.        ]
# on 10,000 with 2 PCs --> total_variance_explained =======================================>  [0.53526158 1.        ]
#


K = np.argmax(total_variance_explained>0.95)+1

# compute the principal components
V = SVD.V
U = U.rows.map(lambda x: (x.index, x.vector[0:K]*S[0:K]))
princ_comps = np.array(list(map(lambda x:x[1], sorted(U.collect(), key = lambda x:x[0]))))



#
# # plot it later!!!!!!!!!!!!
# setosa = princ_comps[iris_target==0]
# versicolor = princ_comps[iris_target==1]
# verginica = princ_comps[iris_target==2]
# plt.scatter(setosa[:,0], setosa[:,1], c="b",label="setosa")
# plt.scatter(versicolor[:,0], versicolor[:,1], c="g",label="versicolor")
# plt.scatter(verginica[:,0], verginica[:,1], c="r",label="verginica")
#





#
# spark = SparkSession.Builder().getOrCreate()
#
#
# # ------------
# from pyspark.mllib.linalg import Vectors
# from pyspark.mllib.linalg.distributed import RowMatrix
#
# rows = spark.sparkContext.parallelize([
#     Vectors.sparse(5, {1: 1.0, 3: 7.0}),
#     Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
#     Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
# ])
#
# for item in rows.collect():
#     print(item)
#
# mat = RowMatrix(rows)
# # Compute the top 4 principal components.
# # Principal components are stored in a local dense matrix.
# pc = mat.computePrincipalComponents(1)
#
# --> computePrincipalComponents(k)[source]
# --> Computes the k principal components of the given row matrix
#
# --> Note This cannot be computed on matrices with more than 65535 columns.
#
#
# # Project the rows to the linear space spanned by the top 4 principal components.
# projected = mat.multiply(pc)
#
# for item in projected.rows.collect():
#     print(item)
#
# # print(pc.variance())
# print("WHAAT")
# print(pc)
#
#
#
#
# # def variance_explained(data, k=1):
# #     """Calculate the fraction of variance explained by the top `k` eigenvectors.
# #
# #     Args:
# #         data (RDD of np.ndarray): An RDD that contains NumPy arrays which store the
# #             features for an observation.
# #         k: The number of principal components to consider.
# #
# #     Returns:
# #         float: A number between 0 and 1 representing the percentage of variance explained
# #             by the top `k` eigenvectors.
# #     """
# #     components, scores, eigenvalues = pca(data, k)
# #
# #     return eigenvalues[:k].sum() * 1. / eigenvalues.sum()
#
#
# # https://stackoverflow.com/questions/33428589/pyspark-and-pca-how-can-i-extract-the-eigenvectors-of-this-pca-how-can-i-calcu?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
#
# from pyspark.ml.feature import VectorAssembler
# #assembler = VectorAssembler(inputCols=<columns of your original dataframe>, outputCol="features")
# #df = assembler.transform(<your original dataframe>).select("features")
#
# from pyspark.sql import SQLContext
# sqlContext = SQLContext(spark.sparkContext)
#
# df = sqlContext.createDataFrame(
#     [
#         (Vectors.dense([1, 2, 0]),),
#         (Vectors.dense([2, 0, 1]),),
#         (Vectors.dense([0, 1, 0]),)
#     ],
#     ("features", )
# )
#
#
# from pyspark.ml.feature import PCA
# pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
# model = pca.fit(df)
# x = sum(model.explainedVariance)
#
# print(x)
# print(model.explainedVariance)
#
#
# # https://stats.stackexchange.com/questions/158801/can-pca-explained-variance-be-computed-from-the-components-or-from-svd-matrices?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
#
#
#
# # https://www.nodalpoint.com/pca-in-spark-1-5/
#
#
