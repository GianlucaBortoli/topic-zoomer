#!/usr/bin/python3
from collections import namedtuple
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.mllib.clustering import LDA
from utils import *
import csv, argparse, time
import shlex, subprocess

def compute(sc, topLeft, bottomRight, step, datasetPath, k, gfs):
    sqlContext = SQLContext(sc)
    data = sc.textFile(datasetPath)
    data = data.mapPartitions(lambda x: csv.reader(x))
    header = data.first()
    data = data.filter(lambda x: x != header)
    result_to_write = []
    res_computation = []
    step = check_step(topLeft, bottomRight, step)
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
        newDocDF_eng = StopWordsRemover(inputCol="words", outputCol="filtered_eng"). \
            transform(docDF)
        newDocDF_eng = newDocDF_eng.drop('words')
        StopWordsRemover.loadDefaultStopWords('italian')
        newDocDF_ita = StopWordsRemover(inputCol="filtered_eng", outputCol="filtered_ita"). \
            transform(newDocDF_eng)
        newDocDF_ita = newDocDF_ita.drop('filtered_eng')
        StopWordsRemover.loadDefaultStopWords('german')
        newDocDF_ger = StopWordsRemover(inputCol="filtered_ita", outputCol="filtered_ger"). \
            transform(newDocDF_ita)
        newDocDF_ger= newDocDF_ger.drop('filtered_ita')

        model = CountVectorizer(inputCol="filtered_ger", outputCol="vectors"). \
            fit(newDocDF_ger)
        result = model.transform(newDocDF_ger)
        corpus = result.select("idd", "vectors").rdd.map(create_corpus).cache()
        # cluster the documents into the k topics using LDA
        ldaModel = LDA.train(corpus, k=k, maxIterations=100, optimizer='online')
        vocabArray = model.vocabulary
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
        s = "; "
        res = "{}, {}, {}, {}, {}".format(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y, s.join(topics_label))
        result_to_write.append(res)
        res_computation.append(topics_label)

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
        copyRecBucketCmd = 'gsutil cp -r {} {}'.format(recFileFolder, gfs_output_path_hdfs)
        copyHdfsRes = subprocess.call(shlex.split(copyHdfsCmd))
        copyBucketRes = subprocess.call(shlex.split(copyBucketCmd))
        copyRecBucketRes = subprocess.call(shlex.split(copyRecBucketCmd))
        # some exit code checks
        if copyBucketRes or copyHdfsRes or copyRecBucketRes:
            print('hdfsRes: {}'.format(copyHdfsRes))
            print('bucketResComp: {}'.format(copyBucketRes))
            print('bucketResRec: {}'.format(copyRecBucketRes))
            print('Something went wrong while copying results')
    return res_computation


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
    parser.add_argument('recomputation', type=int, help='flag to activate recomputation')
    args = parser.parse_args()
    # create area delimeter points
    topLeft = Point(args.tlx, args.tly)
    bottomRight = Point(args.brx, args.bry)
    recomputation = args.recomputation
    print("Top left = ({},{})".format(topLeft.x, topLeft.y))
    print("Bottom right = ({},{})".format(bottomRight.x, bottomRight.y))
    print("Step = {}".format(args.step))
    # check if google filesystem (bucket) is used
    if args.dataset[:3] == "gs:":
        gfs = True
    # create Spark context
    sc = SparkContext(appName="topic_zoomer", pyFiles=["./utils.py"])
    start_isEqual_time = time.time()
    computedSquares = []
    results = []
    if recomputation:
        computedSquares = get_computed_squares()
    print("Computed squares = {}".format(computedSquares))

    if len(computedSquares) != 0:
        # use recomputation
        if not is_equal(topLeft, bottomRight, computedSquares[0]):
            print("not equal")
            for diff in get_diff_squares(topLeft, bottomRight, computedSquares):
                results.append(compute(sc, diff[0], diff[1], args.step, args.dataset, args.k, gfs))
        else:
            end_isEqual_time = time.time()
            print("Time elapsed:", end_isEqual_time - start_isEqual_time)
            print("equal")
    else:
        # do not use recomputation
        results.append(compute(sc, topLeft, bottomRight, args.step, args.dataset, args.k, gfs))

    # if recomputation is not enabled do cleanup before saving results
    if not recomputation:
        rmFolderHdfsCmd = 'hdfs dfs -rm -r -f {}'.format(recFileFolder)
        rmFolderHdfsRes = subprocess.call(shlex.split(rmFolderHdfsCmd))
        res = "{}, {}, {}, {}, ".format(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y)
        # properly format output and save it to HDFS
        tmp = []
        for r in results:
            for r1 in r:
                tmp.extend(r1)
        s = "; "
        res += s.join(tmp)
        final_result = [res]
        to_write = sc.parallelize(final_result)
        to_write.saveAsTextFile(recFileFolder)
        # cleanup if needed
        if rmFolderHdfsRes:
            print('rmFolder: {}'.format(rmFolderHdfsRes))
            print('Something went wrong during cleanup')
        if gfs:
            # copy recomputation stuff from HDFS to local FS
            copyRecFileCmd = 'hdfs dfs -copyToLocal {} /tmp'.format(recFileFolder)
            copyRecFileRes = subprocess.call(shlex.split(copyRecFileCmd))
            copyRecBucketCmd = 'gsutil cp -r {} {}'.format(recFileFolder, gfs_output_path_hdfs)
            copyRecBucketRes = subprocess.call(shlex.split(copyRecBucketCmd))
            # some exit code checks
            if copyRecBucketRes:
                print('bucketResRec: {}'.format(copyRecBucketRes))
                print('Something went wrong copying recomputation files to bucket')
