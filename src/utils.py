#!/usr/bin/python3
from pyspark.mllib.linalg import DenseVector
from pyspark.sql import Row
from collections import namedtuple
import re, csv, os, shlex
import subprocess, shutil

# global vars
Point = namedtuple('Point', ['x', 'y'])
reg = r"[^a-zA-Z| 0-9 | \']"
reg_compiled = re.compile(reg)
gfs_output_path_hdfs = "gs://topic-zoomer/results/"
recFileFolder = "/tmp/Topic_Zoomer_recomputation"


def check_step(topLeft, bottomRight, step):
    return min(step, topLeft.y-bottomRight.y, bottomRight.x - topLeft.x)


def get_computed_squares():
    result = []
    recFileName = "/tmp/Topic_Zoomer_recomputation/recomputation.txt"
    shutil.rmtree(recFileFolder, ignore_errors=True)
    if os.path.isfile(recFileName):
        os.remove(recFileName)
    copyRecFileCmd = 'hdfs dfs -copyToLocal {} /tmp'.format(recFileFolder)
    copyRecFileRes = subprocess.call(shlex.split(copyRecFileCmd))
    mergeFileCmd = 'cat {}/* > {}'.format(recFileFolder, recFileName)
    mergeFileRes = subprocess.call(mergeFileCmd, shell=True)
    if copyRecFileRes or mergeFileRes:
        print('CopyRes: {}'.format(copyRecFileRes))
        print('MergeRes: {}'.format(mergeFileRes))
        print('Something went wrong while copying results')
        return result
    with open(recFileName, "r") as res:
        csvRes = csv.reader(res)
        for row in csvRes:
            tl = Point(x=float(row[0]), y=float(row[1]))
            br = Point(x=float(row[2]), y=float(row[3]))
            result.append([tl, br])
    return result


def get_square_points(tl,br):
    bl = Point(tl.x, br.y)
    tr = Point(br.x, tl.y)
    return [tl, tr, bl, br]


def is_equal(inputTl, inputBr, computedSquares):
    return inputTl.x == computedSquares[0].x and inputTl.y == computedSquares[0].y and \
        inputBr.x == computedSquares[1].x and inputBr.y == computedSquares[1].y


def get_diff_squares(inputTl, inputBr, computedSquares):
    oldSquares = []
    output = []
    inputSquare = get_square_points(inputTl, inputBr)
    common = get_common_squares(inputTl, inputBr, computedSquares)

    for s in computedSquares:
        oldSquares.append(get_square_points(s[0], s[1]))
    for oldS in oldSquares:
        oldSTlBr = [oldS[0], oldS[3]]
        for c in common:
            if point_inside_square(inputBr.x, inputBr.y, oldSTlBr):
                tlOut1 = inputTl
                brOut1 = Point(inputBr.x, c[0].y)
                tlOut2 = Point(inputTl.x, c[0].y)
                brOut2 = Point(c[0].x, inputBr.y)
                output.append([tlOut1, brOut1])
                output.append([tlOut2, brOut2])
            elif point_inside_square(inputTl.x, inputTl.y, oldSTlBr):
                tlOut1 = Point(c[1].x, c[0].y)
                brOut1 = Point(inputBr.x, c[1].y)
                tlOut2 = Point(inputTl.x, c[1].y)
                brOut2 = inputBr
                output.append([tlOut1, brOut1])
                output.append([tlOut2, brOut2])
            elif point_inside_square(inputSquare[2].x, inputSquare[2].y, oldSTlBr):
                tlOut1 = inputTl
                brOut1 = Point(inputBr.x, c[0].y)
                tlOut2 = Point(c[1].x, c[0].y)
                brOut2 = inputBr
                output.append([tlOut1, brOut1])
                output.append([tlOut2, brOut2])
            elif point_inside_square(inputSquare[1].x, inputSquare[1].y, oldSTlBr):
                tlOut1 = inputTl
                brOut1 = Point(c[0].c, c[1].y)
                tlOut2 = Point(inputTl.x, c[1].y)
                brOut2 = inputBr
                output.append([tlOut1, brOut1])
                output.append([tlOut2, brOut2])
            else:
                print("Something gone wrong in diff")
        return output


def get_common_squares(inputTl, inputBr, computedSquares):
    output = []
    oldSquares = []
    inputSquare = get_square_points(inputTl, inputBr)
    for s in computedSquares:
        oldSquares.append(get_square_points(s[0], s[1]))

    for oldS in oldSquares:
        oldSTlBr = [oldS[0], oldS[3]]
        if point_inside_square(inputBr.x, inputBr.y, oldSTlBr):
            tlOut = oldS[0]
            brOut = inputBr
            output.append([tlOut, brOut])
        elif point_inside_square(inputTl.x, inputTl.y, oldSTlBr):
            tlOut = inputTl
            brOut = oldS[3]
            output.append([tlOut, brOut])
        elif point_inside_square(inputSquare[2].x, inputSquare[2].y, oldSTlBr):
            tlOut = Point(inputTl.x, oldS[0].y)
            brOut = Point(oldS[3].x, inputBr.y)
            output.append([tlOut, brOut])
        elif point_inside_square(inputSquare[1].x, inputSquare[1].y, oldSTlBr):
            tlOut = Point(oldS[0].x, inputTl.y)
            brOut = Point(inputBr.x, oldS[3].y)
            output.append([tlOut, brOut])
        else:
            print("Something gone wrong in common")
    return output


def topic_render(x, word_numbers, vocab_array):  
    # specify vector id of words to actual words
    terms = x[0]
    #prob = x[1]
    result = []
    for i in range(word_numbers):
        term = vocab_array[terms[i]]
        result.append(term)
    return result


def remove_punctuation(row):
    return row[0], reg_compiled.sub('', row[1])


def create_corpus(x):
    return [x[0], DenseVector(x[1].toArray())]


def is_inside(row, topLeft, bottomRight, step, squares):
    x = float(row[0])
    y = float(row[1])
    text = row[2]
    idx = 0

    if topLeft.x <= x and topLeft.y >= y and \
        bottomRight.x >= x and bottomRight.y <= y:
        # now I'm inside the selected area
        # range among all the possible little squares
        for s in squares:
            if point_inside_square(x, y, s):
                return (idx, text)
            else:
                idx += 1


def split_string_into_array(row):
    return row[0], row[1].lower().strip().split(" ")


def remove_empty_array(array):
    return array[0], array[1][0] != ''


def create_row(array):
    id = array[0]
    words = array[1]
    return id, Row(idd=id, words=words)


# Determine if a point is inside a given square or not
def point_inside_square(x, y, square):
    topLeft = square[0]
    bottomRight = square[1]

    if topLeft.x <= x and topLeft.y >= y and \
        bottomRight.x >= x and bottomRight.y <= y:
        return True
    else:
        return False


# topLeft and bottomRight are named tuples
def get_squares(topLeft, bottomRight, step):
    # every little square is defined as topLeft and
    # bottomRight angles (as namedtuples)
    Point = namedtuple('Point', ['x', 'y'])
    out = []

    yMin = topLeft.y
    yMax = topLeft.y - step

    while yMax >= bottomRight.y:
        xMin = topLeft.x
        xMax = topLeft.x + step
        while xMax <= bottomRight.x:
            square = [Point(xMin, yMin), Point(xMax, yMax)]
            out.append(square)
            # update x boundaries
            xMin = xMax
            xMax += step
        # update y boundaries
        yMin = yMax
        yMax -= step
    return out
