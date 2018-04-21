# -*- coding: utf-8 -*-


from pandas import json

from pyspark.sql import SparkSession

filename = '/Users/nicolasg-chausseau/Downloads/yelp_dataset/review_1M.json'
# filename = '../data/review_1M.json'
# filename = '/Users/nicolasg-chausseau/Downloads/yelp_dataset/review_MTL_ONLY.json'
# filename = '/Users/nicolasg-chausseau/big_data_project_yelp/data/review_truncated_RAW.json'



# # spark = SparkSession.Builder().getOrCreate()
# # rdd = spark.read.json(filename).limit(1000000).rdd # datapath+'/data/review_trunca®ted_RAW.json'
# json_data=open(filename).read()
# data = json.loads(json_data)
# # pprint(data)




# import numpy as np
# import itertools
# with open(filename) as f_in:
#     x = np.genfromtxt(itertools.islice(f_in, 1, 12, None), dtype='unicode')
#     print(x[0,:])



lines = []
counter = 0
filename_counter = 0
with open(filename) as f:
    for line in f:
        counter += 1
        lines += [line]
        if counter > 50000 - 1:
            outF = open("review_50K_"+str(filename_counter)+".json", "w")
            outF.writelines(lines)
            outF.close()
            filename_counter += 1
            counter = 0
            lines = []


