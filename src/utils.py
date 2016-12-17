#!/usr/bin/python3
from pyspark.mllib.linalg import DenseVector
from pyspark.sql import Row
from collections import namedtuple
import re

# global vars
reg = r"[^a-zA-Z| 0-9 | \']"
reg_compiled = re.compile(reg)


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
    x = int(row[0])
    y = int(row[1])
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
