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


def is_inside(row, topLeft, bottomRight):
    x = int(row[0])
    y = int(row[1])
    text = row[2]

    if topLeft.x <= x and topLeft.y >= y and \
        bottomRight.x >= x and bottomRight.y <= y:
        return text

def split_string_into_array(row):
    return row.lower().strip().split(" ")


def remove_empty_array(array):
    return array[0] != ''


def create_row(array):
    return Row(idd=array[1], words=array[0])
