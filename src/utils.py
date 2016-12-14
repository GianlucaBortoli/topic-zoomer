#!/usr/bin/python3
from pyspark.mllib.linalg import DenseVector
from pyspark.sql import Row
import re

# global vars
reg = r"[^a-zA-Z| 0-9 | \']"
reg_compiled = re.compile(reg)


def topic_render(x, word_numbers, vocab_array):  
    # specify vector id of words to actual words
    terms = x[0]
    prob = x[1]
    result = []
    for i in range(word_numbers):
        term = vocab_array[terms[i]]
        result.append((term, prob[i]))
    return result


def remove_punctuation(text):
    return reg_compiled.sub('', text)


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
            if point_inside_polygon(x, y, s):
                return (idx, text)
            else:
                idx += 1

def split_string_into_array(row):
    return row.lower().strip().split(" ")


def remove_empty_array(array):
    return array[0] != ''


def create_row(array):
    return Row(idd=array[1], words=array[0])


# Determine if a point is inside a given polygon or not
# Polygon is a list of (x,y) pairs.
def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]

    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def get_squares(topLeft, bottomRight, step):
    # every little square is defined as the list of (x,y)
    # of its angles
    out = []

    xMin = topLeft.x
    xMax = topLeft.x + step
    yMin = topLeft.y
    yMax = topLeft.y - step

    while (yMax >= bottomRight.y):
        while (xMax <= bottomRight.x):
            square = [(xMin, yMin), (xMax, yMin), (xMin, yMax), (xMax, yMax)]
            out.append(square)
            # update x boundaries
            xMin = xMax
            xMax += step
        # update y boundaries
        yMin = yMax
        yMax -= step

    return out