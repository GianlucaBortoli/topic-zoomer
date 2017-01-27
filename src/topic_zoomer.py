#!/usr/bin/python3
from collections import namedtuple
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.mllib.clustering import LDA
from utils import *
import csv, os, sys, argparse, logging, time
import shlex, subprocess

# global variables
gfs_output_path_hdfs = "gs://topic-zoomer/results/"


def compute(sc, topLeft, bottomRight, step, datasetPath, k, gfs):
    sqlContext = SQLContext(sc)
    data = sc.textFile(datasetPath)
    data = data.mapPartitions(lambda x: csv.reader(x))
    header = data.first()
    data = data.filter(lambda x: x != header)
    result_to_write = []
    squares = get_squares(topLeft, bottomRight, step)
    # start computing elapsed time here
    start_time = time.time()
    data = data.map(lambda x: is_inside(x, topLeft, bottomRight, step, squares)). \
        filter(lambda x: x is not None)

    data = data.map(remove_punctuation). \
        map(split_string_into_array). \
        filter(remove_empty_array). \
        map(create_row). \
        groupByKey(). \
        map(lambda x : (x[0], list(x[1])))
    # create the dataframes
    allDf = []
    for df in data.collect():
        if df:
            allDf.append([df[0], sqlContext.createDataFrame(df[1])])
    for docDFs in allDf:
        docDF = docDFs[1]
        squareId = docDFs[0]
        StopWordsRemover.loadDefaultStopWords('english')
        newDocDF = StopWordsRemover(inputCol="words", outputCol="filtered"). \
            transform(docDF)
        model = CountVectorizer(inputCol="filtered", outputCol="vectors"). \
            fit(newDocDF)
        result = model.transform(newDocDF)
        corpus = result.select("idd", "vectors").rdd.map(create_corpus).cache()
        # cluster the documents into the k topics using LDA
        ldaModel = LDA.train(corpus, k=k, maxIterations=100, optimizer='online')
        vocabArray = model.vocabulary
        print("Learned topics over vocab of {} words".format(ldaModel.vocabSize()))
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
        result_to_write.append((squareId, topics_label))
    end_time = time.time()
    elapsed_time = end_time - start_time
    result_to_write.append(elapsed_time)
    to_write = sc.parallelize(result_to_write)
    # get dataset size from file name
    size = datasetPath.split('.')[0].split('_')[1]
    if gfs:
        output_folder = "/tmp/Topic_Zoomer_" + str(time.ctime(start_time)).replace(' ', '_').replace(':', '-') + '_' + size
    else:
        output_folder = "Topic_Zoomer_" + str(time.ctime(start_time)).replace(' ', '_').replace(':', '-') + '_' + size

    to_write.saveAsTextFile(output_folder)

    if gfs:
        copyHdfsCmd = 'hdfs dfs -copyToLocal {} {}'.format(output_folder, output_folder)
        copyBucketCmd = 'gsutil cp -r {} {}'.format(output_folder, gfs_output_path_hdfs)
        copyHdfsRes = subprocess.call(shlex.split(copyHdfsCmd))
        copyBucketRes = subprocess.call(shlex.split(copyBucketCmd))
        # some exit code checks
        if copyBucketRes or copyHdfsRes:
            print('hdfsRes: {}'.format(copyHdfsRes))
            print('bucketRes: {}'.format(copyBucketRes))
            print('Something went wrong while copying results')

if __name__ == '__main__':
    # NOTE: env variable SPARK_HOME has to be set in advance
    # The check on the number of parameters is done automatically
    # by the argparse package
    gfs = False
    # command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('k', type=int, help='the number of topics to be found')
    parser.add_argument('tlx', type=float, help='top left point x coordinate')
    parser.add_argument('tly', type=float, help='top left point y coordinate')
    parser.add_argument('brx', type=float, help='bottom right point x coordinate')
    parser.add_argument('bry', type=float, help='bottom right point y coordinate')
    parser.add_argument('step', type=float, help='the square side size')
    parser.add_argument('dataset', type=str, help='the path to the input dataset')
    args = parser.parse_args()
    # create area delimeter points
    Point = namedtuple('Point', ['x', 'y'])
    topLeft = Point(args.tlx, args.tly)
    bottomRight = Point(args.brx, args.bry)
    print("Top left = ({},{})".format(topLeft.x, topLeft.y))
    print("Bottom right = ({},{})".format(bottomRight.x, bottomRight.y))
    print("Step = {}".format(args.step))
    if args.dataset[:3] == "gs:":
        gfs = True
    sc = SparkContext(appName="topic_zoomer", pyFiles=["./utils.py"])
    compute(sc, topLeft, bottomRight, args.step, args.dataset, args.k, gfs)

