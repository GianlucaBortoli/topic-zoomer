#!/usr/bin/env python
import matplotlib.pyplot as plt
import argparse, sys, os, csv
import math

def backToTop(csvFile, reader):
    csvFile.seek(0)
    next(reader, None)
    return

def computePoints(csvFile, reader):
    x = [] # the dataset ids (ie. portion wrt whole)
    y = [] # timings (in mins)
    low = [] # min value for error
    top = [] # max value for error
    # find the dataset parts
    for row in reader:
        _id   = int(row[0])
        _time = float(row[1])
        if _id not in x:
            x.append(_id)
    # compute avg for each dataset id
    for item in x:
        backToTop(csvFile, reader)
        timings = []
        for row in reader:
            _id = int(row[0])
            _time = float(row[1])
            if item == _id:
                timings.append(_time)
        # put average for each id as y coordinate
        avg = sum(timings)/len(timings)
        y.append(math.log(avg/60))
    # compute min & max error
    for item in x:
        backToTop(csvFile, reader)
        tmp_min = tmp_max = None

        for row in reader:
            _id = int(row[0])
            _time = float(row[1])
            if item == _id:
                if tmp_min is None or _time < tmp_min:
                    # update to new min
                    tmp_min = _time
                if tmp_max is None or _time > tmp_max:
                    # update to new max
                    tmp_max = _time
        low.append(math.log(tmp_min/60))
        top.append(math.log(tmp_max/60))
    return x, y, low, top

def main(inFile, outFileName):
    print("Input: {}".format(inFile))
    print("Output: {}.png".format(outFileName))
    # handle input file generic errors
    if not os.path.isfile(inFile):
        print('File "{}" does not exists'.format(inFile))
        sys.exit(1)
    # deal with csv
    with open(inFile, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        # skip header
        next(reader, None)
        x, y, low, top = computePoints(csvfile, reader)
    print("x, y", x, y)
    print("best/worst cases", low, top)
    # compute deltas between avg and min/max values
    # if a sequence of shape 2xN, errorbars are drawn at -row1 and +row2 relative to the data
    diffs_x = []
    diffs_y = []
    for a in y:
        diffs_x.append(abs(low[y.index(a)] - a))
        diffs_y.append(abs(top[y.index(a)] - a))
    print("diffs", diffs_x, diffs_y)
    # create figure & save it
    plt.errorbar(x, y, yerr=[diffs_x, diffs_y], ecolor='red', elinewidth=1, fmt='o-')
    plt.xticks(x, x)
    #plt.xlabel('% of the dataset')
    #plt.xlabel('Step')
    plt.xlabel('% of common area')
    plt.ylabel('Time (min)')
    #axes = plt.gca()
    #axes.set_xlim([3, 26])
    plt.savefig(outFileName + '.png')
    print("Done!")


if __name__ == '__main__':
    # NOTE: csv's rows _must_ be ordered incrementally wrt id
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str,
        help='the input file with data to be displayed',
        required=True)
    parser.add_argument('--output', type=str,
        help='the output file name for the chart (wo extension)',
        required=True)
    args = parser.parse_args()
    main(os.path.abspath(args.input),
        os.path.abspath(args.output))
