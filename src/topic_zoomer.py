#!/usr/bin/python3
from collections import namedtuple
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.mllib.clustering import LDA
from utils import *
import csv, os, sys, argparse, logging

def compute(topLeft, bottomRight, step, datasetPath, master):
    # TODO: implement step stuff
    if master is None:
        master = "local[2]"

    sc = SparkContext(appName="topic_zoomer", 
        master=master,
        pyFiles=["./utils.py"])
    #sc.addPyFile("./utils.py")

    sqlContext = SQLContext(sc)
    data = sc.textFile(datasetPath)
    data = data.mapPartitions(lambda x: csv.reader(x))
    header = data.first()
    data = data.filter(lambda x: x != header)
    data = data.map(lambda x: is_inside(x, topLeft, bottomRight)).\
        filter(lambda x: x is not None)
    data = data.map(remove_punctuation).\
        map(split_string_into_array).\
        filter(remove_empty_array).\
        zipWithIndex().\
        map(create_row)
    docDF = sqlContext.createDataFrame(data)
    StopWordsRemover.loadDefaultStopWords('english')
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    newDocDF = remover.transform(docDF)
    Vector = CountVectorizer(inputCol="filtered", outputCol="vectors")
    model = Vector.fit(newDocDF)
    result = model.transform(newDocDF)
    corpus_size = result.count()  # total number of words
    corpus = result.select("idd", "vectors").rdd.map(create_corpus).cache()
    # cluster the documents into three topics using LDA
    ldaModel = LDA.train(corpus, k=5, maxIterations=100, optimizer='online')
    topics = ldaModel.topicsMatrix()
    vocabArray = model.vocabulary
    print("Learned topics over vocab of {} words".format(ldaModel.vocabSize()))
    wordNumbers = 10  # number of words per topic
    topicIndices = sc.parallelize(ldaModel.describeTopics(maxTermsPerTopic=wordNumbers))
    topics_final = topicIndices.map(lambda x: topic_render(x, wordNumbers, vocabArray)).collect()
    # compute labels
    topics_label = []
    for topic in topics_final:
        for topic_term in topic:
            if topic_term[0] not in topics_label:
                topics_label.append(topic_term[0])
                break
    # print topics
    for topic in range(len(topics_final)):
        print("Topic " + str(topics_label[topic]))
        for term in topics_final[topic]:
            print('\t', term)


if __name__ == '__main__':
    logging.basicConfig(filename='topic_zoomer.log', format='%(levelname)s: %(message)s', level=logging.INFO)
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparkhome', type=str, help='the SPARK_HOME path. Overrides systemwhide env variable')
    parser.add_argument('--master', type=str, help='the master url')
    required = parser.add_argument_group('required named arguments')
    required.add_argument('--tlx', type=float, help='top left point x coordinate', required=True)
    required.add_argument('--tly', type=float, help='top left point y coordinate', required=True)
    required.add_argument('--brx', type=float, help='bottom right point x coordinate', required=True)
    required.add_argument('--bry', type=float, help='bottom right point y coordinate', required=True)
    required.add_argument('--step', type=float, help='the square side size', required=True)
    required.add_argument('--dataset', type=str, help='the path to the input dataset', required=True)
    args = parser.parse_args()
    # some error checks on input args
    if args.sparkhome == None:
        logging.info("Using system SPARK_HOME")
    else:
        logging.info("Setting SPARK_HOME to {}".format(args.sparkhome))
        os.environ['SPARK_HOME'] = args.sparkhome
    # create area delimeter points
    Point = namedtuple('Point', ['x', 'y'])
    topLeft = Point(args.tlx, args.tly)
    bottomRight = Point(args.brx, args.bry)
    compute(topLeft, bottomRight, args.step, args.dataset, args.master)