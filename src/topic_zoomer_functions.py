from pyspark.mllib.linalg import DenseVector
from pyspark.sql import Row
import re
reg = r"[^a-zA-Z| 0-9 | \']"
reg_compiled = re.compile(reg)


def topic_render(x, word_numbers, vocab_array):  # specify vector id of words to actual words
    terms = x[0]
    prob = x[1]
    result = []
    for i in range(word_numbers):
        term = vocab_array[terms[i]]
        result.append((term, prob[i]))
    return result


def remove_punctuation(text):  # remove punctuation
    return reg_compiled.sub('', text)


def create_corpus(x):
    return [x[0], DenseVector(x[1].toArray())]


def check_area(row, p1, p2, p3, p4):
    x = int(row[0])
    y = int(row[1])
    text = row[2]
    if p1.y >= y and p2.y >= y and \
                    p3.y <= y and p4.y <= y and \
                    p1.x <= x and p2.x >= x and \
                    p3.x <= x and p4.x >= x:
        return text


def split_string_into_array(row):
    return row.lower().strip().split(" ")


def remove_empty_array(array):
    return array[0] != ''


def create_row(array):
    return Row(idd=array[1], words=array[0])

