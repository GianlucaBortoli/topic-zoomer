from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import CountVectorizer
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors, DenseVector
from pyspark.ml.feature import StopWordsRemover
from pyspark import SparkContext
from collections import namedtuple
import re
import csv
import os
os.environ['SPARK_HOME'] = "/home/pier/BigData/spark"

reg = r"[^a-zA-Z| 0-9 | \']"
reg_compiled = re.compile(reg)
point = namedtuple('Point', ['x', 'y'])
p1 = point(0, 100)
p2 = point(100, 100)
p3 = point(0, 0)
p4 = point(100, 0)



def remove_punctuation(text):  # remove punctuation
    return reg_compiled.sub('', text)


def topic_render(x):  # specify vector id of words to actual words
    terms = x[0]
    prob = x[1]
    result = []
    for i in range(wordNumbers):
        term = vocabArray[terms[i]]
        result.append((term, prob[i]))
    return result


def check_area(row):
    x = int(row[0])
    y = int(row[1])
    text = row[2]
    if p1.y>=y and p2.y>=y and \
                    p3.y <= y and p4.y <= y and \
                    p1.x<=x and p2.x >= x and \
                    p3.x<=x and p4.x >= x:
        return text


def split_string_into_array(row):
    return row.lower().strip().split(" ")


def remove_empty_array(array):
    return array[0] != ''


def create_row(array):
    return Row(idd=array[1], words=array[0])


# Load and parse the data
sc = SparkContext(appName="Project", master="spark://localhost.localdomain:7077")
sqlContext = SQLContext(sc)
path = "input/synthetic_dataset.csv"
data = sc.textFile(path)
data = data.mapPartitions(lambda x: csv.reader(x))
header = data.first()
data = data.filter(lambda x: x != header)
data = data.map(check_area).filter(lambda x: x is not None)
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
corpus = result.select("idd", "vectors").rdd.map(lambda x: [x[0], DenseVector(x[1].toArray())]).cache()
# Cluster the documents into three topics using LDA
ldaModel = LDA.train(corpus, k=5, maxIterations=100, optimizer='online')
topics = ldaModel.topicsMatrix()
vocabArray = model.vocabulary
print("Learned topics (as distributions over vocab of {} words".format(ldaModel.vocabSize()))
wordNumbers = 10  # number of words per topic
topicIndices = sc.parallelize(ldaModel.describeTopics(maxTermsPerTopic=wordNumbers))
topics_final = topicIndices.map(lambda x: topic_render(x)).collect()
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
