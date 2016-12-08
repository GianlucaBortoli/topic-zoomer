import csv
import os
from collections import namedtuple
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.mllib.clustering import LDA

from topic_zoomer_functions import *

os.environ['SPARK_HOME'] = "/home/pier/BigData/spark"

point = namedtuple('Point', ['x', 'y'])
p1 = point(0, 100)
p2 = point(100, 100)
p3 = point(0, 0)
p4 = point(100, 0)

# Load and parse the data
sc = SparkContext(appName="Project", master="spark://localhost.localdomain:7077",
                  pyFiles=["src/topic_zoomer_functions.py"])
sc.addPyFile("src/topic_zoomer_functions.py")
sqlContext = SQLContext(sc)
path = "input/synthetic_dataset.csv"
data = sc.textFile(path)
data = data.mapPartitions(lambda x: csv.reader(x))
header = data.first()
data = data.filter(lambda x: x != header)
data = data.map(lambda x: check_area(x, p1, p2, p3, p4)).filter(lambda x: x is not None)
data = data.map(remove_punctuation).map(split_string_into_array).filter(remove_empty_array).\
    zipWithIndex().map(create_row)
docDF = sqlContext.createDataFrame(data)
StopWordsRemover.loadDefaultStopWords('english')
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
newDocDF = remover.transform(docDF)
Vector = CountVectorizer(inputCol="filtered", outputCol="vectors")
model = Vector.fit(newDocDF)
result = model.transform(newDocDF)
corpus_size = result.count()  # total number of words
corpus = result.select("idd", "vectors").rdd.map(create_corpus).cache()
# Cluster the documents into three topics using LDA
ldaModel = LDA.train(corpus, k=5, maxIterations=100, optimizer='online')
topics = ldaModel.topicsMatrix()
vocabArray = model.vocabulary
print("Learned topics (as distributions over vocab of {} words".format(ldaModel.vocabSize()))
wordNumbers = 10  # number of words per topic
topicIndices = sc.parallelize(ldaModel.describeTopics(maxTermsPerTopic=wordNumbers))
topics_final = topicIndices.map(lambda x: topic_render(x, wordNumbers, vocabArray)).collect()
topics_label = []
for topic in topics_final:
    for topic_term in topic:
        if topic_term[0] not in topics_label:
            topics_label.append(topic_term[0])
            break


# Print topics
for topic in range(len(topics_final)):
    print("Topic " + str(topics_label[topic]))
    for term in topics_final[topic]:
        print('\t', term)
