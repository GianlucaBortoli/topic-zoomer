from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import CountVectorizer
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors, DenseVector
from pyspark.ml.feature import StopWordsRemover
from pyspark import SparkContext
import re

reg = r"[^a-zA-Z| 0-9 | \']"
reg_compiled = re.compile(reg)


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


# Load and parse the data
sc = SparkContext(appName="Project", master="spark://localhost.localdomain:7077")

sqlContext = SQLContext(sc)
path = "input/test_m.txt"
data = sc.textFile(path).map(remove_punctuation).map(lambda x: x.lower().strip().split(" ")).filter(
    lambda x: x[0] != '').zipWithIndex().map(lambda x: Row(idd=x[1], words=x[0]))

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
