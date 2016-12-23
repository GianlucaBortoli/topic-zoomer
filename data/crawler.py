#!/usr/bin/env
from multiprocessing.pool import ThreadPool
from queue import Queue
from bs4 import BeautifulSoup
import argparse, requests, os, sys
import csv, random

# global vars
concurrent = 100
q = Queue(concurrent * 2)
maxFieldSize = 131072

def downloadPage(url):
    strippedUrl = url.strip()
    try:
        r = requests.get(strippedUrl)
        if (r.status_code == 200):
            print("\t", strippedUrl)
            body = r.text
            soup = BeautifulSoup(body, 'html.parser')
            # remove script & style parts
            for s in soup(["script", "style"]):
                s.extract()
            # extract text from body
            text = soup.get_text()
            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            strippedText = ''.join(chunk for chunk in chunks if chunk)
            # truncate string at max field size allowed in csv python module
            return strippedText[:maxFieldSize]
        else:
            print("Skipping (status code = {})".format(r.status_code))
    except Exception as e:
        print(e)

def main(inputFile, outputFile, _min, _max):
    urls = []
    with open(inputFile) as i:
        urls = i.readlines()
    print('Downloading {} pages...'.format(len(urls)))
    print('Min = ', _min)
    print('Max = ', _max)

    try:
        results = ThreadPool(concurrent).imap_unordered(downloadPage, urls)

        with open(outputFile, 'w+') as o:
            fieldnames = ['x', 'y', 'text']
            writer = csv.DictWriter(o, delimiter=',', fieldnames=fieldnames)
            writer.writeheader()
            for t in results:
                # write text back into csv with fake geotags
                writer.writerow({
                    'x': round(random.uniform(_min, _max), 3),
                    'y': round(random.uniform(_min, _max), 3),
                    'text': t
                })
    except KeyboardInterrupt:
        print("\nStopped by user")
        sys.exit(1)


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='the input file with the URLs', required=True)
    parser.add_argument('--output', type=str, help='the output csv file', required=True)
    parser.add_argument('--min', type=int, help='the min value for generating fake geotags', required=True)
    parser.add_argument('--max', type=int, help='the max value for generating fake geotags', required=True)
    args = parser.parse_args()
    # get abs path from file
    ifilePath = os.path.abspath(args.input)
    ofilePath = os.path.abspath(args.output) 
    main(ifilePath, ofilePath, args.min, args.max)