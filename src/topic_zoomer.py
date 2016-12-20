#!/usr/bin/python3
from collections import namedtuple
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.mllib.clustering import LDA
from utils import *
import csv, os, sys, argparse, logging


def compute(sc, topLeft, bottomRight, step, datasetPath, k):
    sqlContext = SQLContext(sc)
    data = sc.textFile(datasetPath)
    data = data.mapPartitions(lambda x: csv.reader(x))
    header = data.first()
    data = data.filter(lambda x: x != header)

    squares = get_squares(topLeft, bottomRight, step)
    data = data.map(lambda x: is_inside(x, topLeft, bottomRight, step, squares)).\
        filter(lambda x: x is not None)

    data = data.map(remove_punctuation).\
        map(split_string_into_array).\
        filter(remove_empty_array).\
        map(create_row).\
        groupByKey().\
        map(lambda x : (x[0], list(x[1])))
    # create the dataframes
    allDf = []
    for df in data.collect():
        if df:
            allDf.append(sqlContext.createDataFrame(df[1]))
    for docDF in allDf:
        squareId = docDF.select('idd').distinct().collect()
        StopWordsRemover.loadDefaultStopWords('english')
        newDocDF = StopWordsRemover(inputCol="words", outputCol="filtered").\
            transform(docDF)
        model = CountVectorizer(inputCol="filtered", outputCol="vectors").\
            fit(newDocDF)
        result = model.transform(newDocDF)
        corpus_size = result.count()
        corpus = result.select("idd", "vectors").rdd.map(create_corpus).cache()
        # cluster the documents into the k topics using LDA
        ldaModel = LDA.train(corpus, k=k, maxIterations=100, optimizer='online')
        topics = ldaModel.topicsMatrix()
        vocabArray = model.vocabulary
        logging.info("Learned topics over vocab of {} words".format(ldaModel.vocabSize()))
        wordNumbers = 10  # number of words per topic
        topicIndices = sc.parallelize(ldaModel.describeTopics(maxTermsPerTopic=wordNumbers))

        toBePrinted = min(len(vocabArray), wordNumbers)
        topics_final = topicIndices.map(lambda x: topic_render(x, toBePrinted, vocabArray)).collect()
        # compute labels
        topics_label = []
        for topic in topics_final:
            for topic_term in topic:
                if topic_term not in topics_label:
                    topics_label.append(topic_term)
                    break
        # print topics
        for topic in topics_label:
            logging.info("Square: {} -> topic: {}".format(squareId, topic))


if __name__ == '__main__':
    logging.basicConfig(filename='topic_zoomer.log', format='%(levelname)s: %(message)s', level=logging.INFO)
    # command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sparkhome', type=str, help='the SPARK_HOME path')
    parser.add_argument('--master', type=str, help='the master url')
    parser.add_argument('--k', type=int, help='the number of topics to be found', default=5)
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
    logging.info("Top left = ({},{})".format(topLeft.x, topLeft.y))
    logging.info("Bottom right = ({},{})".format(bottomRight.x, bottomRight.y))
    logging.info("Step = {}".format(args.step))

    if args.master is None:
        args.master = "local[2]"

    sc = SparkContext(appName="topic_zoomer",
        master=args.master,
        pyFiles=["./utils.py"])

    compute(sc, topLeft, bottomRight, args.step, args.dataset, args.k)
