import copy
import sys
import os
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer

# from constants import SEED

# # USE THE CODE BELOW TO PRODUCE DATA FILES ONLY FOR MTL.
# # fyi there's this aberrant datapoint to remove, its coordinates are not in Montreal: {"business_id":"UntbR8C0Mxsfd-hNBZXO-w","name":"La Chamade","neighborhood":"","address":"Via San Nullo 48","city":"Montreal","state":"CA","postal_code":"80078","latitude":40.8914165327,"longitude":14.0926355127,"stars":4.5,"review_count":4,"is_open":1,"attributes":{"Alcohol":"beer_and_wine","Ambience":{"casual":false,"classy":false,"hipster":false,"intimate":false,"romantic":false,"touristy":false,"trendy":false,"upscale":false},"BikeParking":true,"BusinessAcceptsCreditCards":true,"BusinessParking":{"garage":false,"lot":false,"street":false,"valet":false,"validated":false},"GoodForKids":true,"HasTV":true,"NoiseLevel":"quiet","OutdoorSeating":true,"RestaurantsAttire":"casual","RestaurantsDelivery":true,"RestaurantsGoodForGroups":true,"RestaurantsPriceRange2":2,"RestaurantsReservations":true,"RestaurantsTableService":true,"RestaurantsTakeOut":true,"WiFi":"no"},"categories":["Pizza","Restaurants"],"hours":{"Friday":"11:30-0:30","Monday":"11:30-0:30","Saturday":"11:30-0:30","Sunday":"11:30-0:30","Thursday":"11:30-0:30","Tuesday":"11:30-0:30","Wednesday":"11:30-0:30"}}
#
# # make a new file: review.json with the city included:
# spark = SparkSession.Builder().getOrCreate()
# filename = '/Users/nicolasg-chausseau/Downloads/yelp_dataset/business.json'
# dataset = spark.read.format('libsvm').json(filename)
# businesses_MTL_ONLY = dataset.select(
#         dataset.business_id,
#         dataset.name, dataset.neighborhood, dataset.address, dataset.city, dataset.state, dataset.postal_code, dataset.latitude, dataset.longitude, dataset.stars, dataset.review_count, dataset.is_open, dataset.attributes, dataset.categories, dataset.hours
#     ).where((dataset.city == "Montreal") & (dataset.name != "La Chamade"))
# # businesses_MTL_ONLY.coalesce(1).write.format('json').save("/Users/nicolasg-chausseau/Downloads/yelp_dataset/business_MTL_ONLY.json")
# businesses_MTL_ONLY.write.format('json').save("/Users/nicolasg-chausseau/Downloads/yelp_dataset/business_MTL_ONLY.json")
# # businesses_MTL_ONLY.write.format('csv').save("/Users/nicolasg-chausseau/Downloads/yelp_dataset/business_MTL_ONLY.csv") # gives error
# # businesses_MTL_ONLY.rdd.saveAsTextFile("/Users/nicolasg-chausseau/Downloads/yelp_dataset/business_MTL_ONLY.csv") # not csv just junk rdd with parentheses.
# # businesses_MTL_ONLY.write.csv("/Users/nicolasg-chausseau/Downloads/yelp_dataset/business_MTL_ONLY.csv", sep=',') # error
# businesses_MTL_ONLY.toPandas().to_csv("/Users/nicolasg-chausseau/Downloads/yelp_dataset/business_MTL_ONLY.csv")
#
#
# print(businesses_MTL_ONLY)
# print("passed")
#
# # join those entries with the review.json: and make a new file review_MTL_ONLY.json:
# filename = '/Users/nicolasg-chausseau/Downloads/yelp_dataset/review.json'
# rdd = spark.read.json(filename).rdd # datapath+'/data/review_truncated_RAW.json'
# print(rdd)
# reviews = spark.createDataFrame(rdd)
# # for item in df.collect():
# #     print(item)
# print(reviews)
#
# review_MTL_ONLY = reviews.join(businesses_MTL_ONLY, reviews.business_id == businesses_MTL_ONLY.business_id, "inner")\
#     .select(
#         reviews.review_id,
#         reviews.user_id,
#         reviews.business_id,
#         reviews.stars,
#         reviews.date,
#         reviews.text,
#         reviews.useful,
#         reviews.funny,
#         reviews.cool,
#         businesses_MTL_ONLY.city
#     )
# print(review_MTL_ONLY)
# # review_MTL_ONLY.coalesce(1).write.format('json').save("/Users/nicolasg-chausseau/Downloads/yelp_dataset/review_MTL_ONLY.json")
# review_MTL_ONLY.write.format('json').save("/Users/nicolasg-chausseau/Downloads/yelp_dataset/review_MTL_ONLY.json")
# print("passed writing reviews to file")







def get_user_business(rating, user_mean, item_mean, rating_global_mean):
  return rating-(user_mean +item_mean-rating_global_mean)

def get_final_ratings(i, user_mean, item_mean, global_average_rating):
  final_ratings = i+user_mean+item_mean-global_average_rating
  return final_ratings

def main():
  spark = SparkSession.Builder().getOrCreate()
  # seed = int(sys.argv[SEED])
  seed = 123#int(sys.argv[1])
  # datapath = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
  # datapath = "/Users/nicolasg-chausseau/big_data_project_yelp"

  # filename = '/Users/nicolasg-chausseau/Downloads/yelp_dataset/review.json'
  filename = 'review_50K_0.json'
  # filename = '/Users/nicolasg-chausseau/Downloads/yelp_dataset/review_MTL_ONLY.json'
  # filename = '/Users/nicolasg-chausseau/big_data_project_yelp/data/review_truncated_RAW.json'

  rdd = spark.read.json(filename).limit(100).rdd # datapath+'/data/review_trunca®ted_RAW.json'
  df = spark.createDataFrame(rdd)
  (training, test) = df.randomSplit([0.8, 0.2], seed)

  # userIdRdd1 = test.select('user_id').rdd.distinct().zipWithIndex().map(lambda x: (x[0][0], x[1]))
  # businessIdRdd1 = test.select('business_id').rdd.distinct().zipWithIndex().map(lambda x: (x[0][0], x[1]))
  userIdRdd1 = test.select('user_id').rdd.distinct().zipWithIndex().map(lambda x: (x[0][0], x[1]))
  businessIdRdd1 = test.select('business_id').rdd.distinct().zipWithIndex().map(lambda x: (x[0][0], x[1]))

  # convert to dataframe
  userIdDf2 = spark.createDataFrame(userIdRdd1)\
                  .withColumnRenamed('_1', 'user_id') \
                  .withColumnRenamed('_2', 'user_id_indexed')
  businessIdDf2 = spark.createDataFrame(businessIdRdd1) \
                      .withColumnRenamed('_1', 'business_id') \
                      .withColumnRenamed('_2', 'business_id_indexed')

  # join user id zipped with index and business id with index
  test = test.join(userIdDf2, ['user_id'], 'left').join(businessIdDf2, ['business_id'], 'left')

  # # get user mean
  user_mean = training.groupBy('user_id').mean('stars').withColumnRenamed('avg(stars)', 'user-mean')

  # --------Nic: some stats---------
  # for item in user_mean.collect():
  #     print(item)
  #
  # # get avg variance, i.e. std, to see if users always give the same reviews:
  # user_mean = training.groupBy('user_id').agg(stddev("stars")).withColumnRenamed('std(stars)', 'user-std')
  # for item in user_mean.collect():
  #     print(item)
  #
  # # get MIN-MAX RATINGS DIFFERENCE, to see if users always give the same reviews:
  # sys.exit()
  # --------/Nic: some stats---------

  # get item mean
  business_mean = training.groupBy('business_id').mean('stars').withColumnRenamed('avg(stars)', 'business-mean')

  # join user mean df and training df
  training = training.join(user_mean, ['user_id']).select(training['user_id'], training['business_id'], training['stars'], user_mean['user-mean'])

  # attempt to adjust with the user-mean ... to center around zero...
  # training.map(lambda x: )

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
  userIdDf = spark.createDataFrame(userIdRdd)\
                  .withColumnRenamed('_1', 'user_id') \
                  .withColumnRenamed('_2', 'user_id_indexed')
  businessIdDf = spark.createDataFrame(businessIdRdd) \
                      .withColumnRenamed('_1', 'business_id') \
                      .withColumnRenamed('_2', 'business_id_indexed')
  # join user id zipped with index and business id with index
  training = training.join(userIdDf, ['user_id'], 'left').join(businessIdDf, ['business_id'], 'left')
  als = ALS(maxIter=5,
            # rank=70,
            rank=3,
            # regParam=0.01,
            regParam=0.01,
            userCol='user_id_indexed',
            itemCol='business_id_indexed',
            ratingCol='user-business-interaction',
            coldStartStrategy='drop')
  als.setSeed(seed)
  model = als.fit(training)

  # Evaluate the model by computing the RMSE on the test data
  predictions = model.transform(test)
  
  predictions = predictions.join(user_mean, ['user_id'],'left')
  predictions = predictions.join(business_mean, ['business_id'], 'left')
  rating_global_mean = training.groupBy().mean('stars').head()[0]
  predictions = predictions.na.fill(rating_global_mean)

  # final_stars = predictions.withColumn('final-stars', get_final_ratings(predictions['prediction'],
  #                                         predictions['user-mean'],
  #                                         predictions['business-mean'],
  #                                         rating_global_mean))

  # get RMSE for baseline: business_mean -->
  # 1.2108445089805764 @1,200 reviews & rank=70
  # 1.4058098693770635   20,000 & rank=70
  # 1.4937988815601055 100,000 & rank=70
  # 1.4058098693770635    100,000 & rank=20
  #  1.4058098693770635   100,000 & rank=3
  final_stars = predictions.withColumn('final-stars', predictions['business-mean'])

  # # # get RMSE for baseline: user_mean -->
  # 1.1380594497335592 @1,200 reviews
  # 1.3111248442800931 @20,000
  # 1.474023035158964 @ 100,000 & rank=70
  # 1.474023035158964 @100,000 & rank=3
  # 1.474023035158964 @ 100,000 & rank=20
  # final_stars = predictions.withColumn('final-stars', predictions['user-mean'])
  # # CONCLUSION: the "easiness" of the users is more predictive of the rating that ALS.
  # # CONCLUSION 1: the dataset inherently is unusable? Can we correct for this, other than just with regularization?

  # # # get RMSE for baseline: avg of user_mean and business_mean --> 1.1419833930002228 @1,200 reviews & rank=70
  # final_stars = predictions.withColumn('final-stars', (predictions['user-mean'] + predictions['business-mean'])/2)

  # # # # normal RMSE: --> 3.8883612142281243 // 3.889945668792359 with reg param=1.0 // 3.887027078929205 with reg param=0.1 @1,200 reviews & rank=70
  # final_stars = predictions.withColumn('final-stars', predictions['prediction'])

  # RMSE with regularisation (should boost score) -->
  # 1.1546294726362072 @1,200 reviews & rank=70
  # 1.2161649258114644 @only montreal reviews. & rank=70
  # 1.4297730289858577 @20,000 reviews & rank=70
  # 1.5648164315827728 @100,000 reviews. & rank=70
  # 1.6506232738462265 @100,000 reviews & rank=3
  #  1.5741020734885307 @ 100,000 & rank=20
  # FROM OTHER SCRIPT:
  #     1.3649280967989137 @10,000 & rank=70 & regParam=0.01
  #     1.5527545323387852  @100,000 & rank=70 & regParam=0.1
  #     1.5532972134819825  @100,000 & rank=70 & regParam=0.01
  #     1.7511860321338941  @100,000 & rank=3 & regParam=0.01
  #     1.5505409385203348  @100,000 & rank=700 & regParam=0.01 & maxIter=20
  #     1.5809661522209637  @100,000 & rank=70 & regParam=0.01 & maxIter=6
  # final_stars = predictions.withColumn('final-stars', get_final_ratings(predictions['prediction'],
  #                                                                       predictions['user-mean'],
  #                                                                       predictions['business-mean'],
  #                                                                       rating_global_mean))


  evaluator = RegressionEvaluator(metricName='rmse',
                                  labelCol='stars',
                                  predictionCol='final-stars')

  rmse = evaluator.evaluate(final_stars)
  print(float(rmse))

  # variance and principal components:
  # data = [(Vectors.dense([0.0, 1.0, 0.0, 7.0, 0.0]),), (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),), (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]

  # predictions.show()
  final_stars.show()
  print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

  #def
  dictOfAllUsers = final_stars.rdd.map(lambda x: (x['user_id'], 1)).reduceByKey(lambda a, b: a+b).collectAsMap()
  for key, value in dictOfAllUsers.items():
      print(str(key), str(value))

  def returnMergedDict(a, b):
    print(a,b)
    dictOfAllUsers_COPY = dict2 = copy.deepcopy(dictOfAllUsers)
    # dictOfAllUsers_COPY[b['user_id']] = b['final-stars']
    dictOfAllUsers_COPY[b[0]] = b[1]  # we know that those users are different every the time
    dictOfAllUsers_COPY[a[0]] = a[1]
    print("**********newdict**********")
    for key, value in dictOfAllUsers_COPY.items():
        print(str(key), str(value))
    return dictOfAllUsers_COPY

  pcaReadyInput = final_stars.rdd.map(lambda x: (x['business_id'], (x['user_id'], x['final-stars']))).reduceByKey(lambda a, b: returnMergedDict(a, b))  # returnMergedDict(x)

  def mergeDicts(x):
    print("hello")
    print(x)
    return 1
  pcaReadyInput = final_stars.rdd.map(lambda x: (x['business_id'], (x['user_id'], x['final-stars']))).reduceByKey(lambda x: mergeDicts(x))  # returnMergedDict(x)

  print(pcaReadyInput)
  for item in pcaReadyInput.collect():
      print(item)

  #pcaReadyInput =

  # df
  # .map(x=> x.getAs[String]("machine") -> (x.getAs[Int]("code"), x.getAs[Int]("code2"),x.getAs[Int]("value")))
  # .groupByKey
  # .mapValues( seq => {
  #     var result = Array.ofDim[Int](256, 256)
  # seq.foreach{ case (i,j,value) => result(i)(j) = value }
  # result
  # })

  #final_stars.groupByKey("business_id").mapValues()









if __name__ == '__main__':
    main()