import sys
from py4j.compat import long
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

spark = SparkSession \
    .builder \
    .appName("basic_als_recommender") \
    .getOrCreate()

# lines = spark.read.text("./data/sample_movielens_ratings.txt").rdd

# df = spark.read.json("/Users/nicolasg-chausseau/big_data_project_yelp/data/review_truncated_VALID_JSON.json")
df = spark.read.json("/Users/nicolasg-chausseau/big_data_project_yelp/data/review_truncated_RAW.json")

# Displays the content of the DataFrame to stdout
# df.show()
print(df.take(10))
for item in df.take(30):
    print(item)
print()


# useful code from LA2 to adapt / reuse:

#
# parts = lines.map(lambda row: row.value.split("::"))
# ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2]), timestamp=long(p[3])))
#
# ratings = spark.createDataFrame(ratingsRDD)
# (training, test) = ratings.randomSplit([0.8, 0.2], sys.argv[1])
#
# global_avg = training.select("rating").groupBy().avg("rating")
# global_avg = global_avg.withColumnRenamed("avg(rating)", "global-mean")
#
# avg_rating_for_each_user = training.select("userId", "rating").groupBy("userId").avg("rating")
# avg_rating_for_each_user = avg_rating_for_each_user.withColumnRenamed("avg(rating)", "user-mean")
#
# avg_rating_for_each_movie = training.select("movieId", "rating").groupBy("movieId").avg("rating")
# avg_rating_for_each_movie = avg_rating_for_each_movie.withColumnRenamed("avg(rating)", "item-mean")
#
# c = training.join(avg_rating_for_each_user, ['userId'])
# d = c.join(avg_rating_for_each_movie, ['movieId'])
# e = d.crossJoin(global_avg) # , , 'cross'
#
# user_item_interaction_udf = udf(lambda rating, user_mean, item_mean, global_mean: rating - (user_mean+item_mean-global_mean), DoubleType())
# f = e.withColumn("user-item-interaction", user_item_interaction_udf(e.rating, e["user-mean"], e['item-mean'], e["global-mean"]))
#
# als = ALS(maxIter=5, rank=70, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="user-item-interaction", coldStartStrategy="drop")
# als.setSeed(int(sys.argv[1]))
# model = als.fit(f)
#
# predictions = model.transform(test)
#
# c2 = predictions.join(avg_rating_for_each_user, ['userId'])
# d2 = c2.join(avg_rating_for_each_movie, ['movieId'])
# e2 = d2.crossJoin(global_avg) # , , 'cross'
#
# adjusted_rating_udf = udf(lambda i, user_mean, item_mean, global_mean: i + user_mean + item_mean - global_mean, DoubleType())
# predictions_final = e2.withColumn("prediction", adjusted_rating_udf(e2["prediction"], e2["user-mean"], e2['item-mean'], e2["global-mean"]))
#
# evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
# rmse = evaluator.evaluate(predictions_final)
# print(str(rmse))
