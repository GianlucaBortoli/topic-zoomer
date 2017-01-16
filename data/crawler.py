#!/usr/bin/env
from multiprocessing import Pool, cpu_count
from bs4 import BeautifulSoup
import argparse, requests, os, sys
import csv, random

# global vars
MAXFIELDSIZE = 131072

def downloadPage(geo_url):
    x = float(geo_url[0])
    y = float(geo_url[1])
    urls = geo_url[2].replace('{', '').replace('}', '').split('|')

    out = []
    for u in urls:
        strippedUrl = u.strip()
        try:
            r = requests.get(strippedUrl)

            if (r.status_code == 200 and 
                'text/html' in r.headers['Content-Type']):
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
                out.append([x, y, strippedText[:MAXFIELDSIZE]])
            else:
                print('Skipping (sc={}, ct={})'.format(r.status_code, r.headers['Content-Type']))
        except Exception as e:
            print(e)
    return out

def main(inputFile, outputFile):    
    urls = []
    geo_urls = []
    with open(inputFile) as i:
        #urls = i.readlines()
        geo_urls = list(csv.reader(i))
    print('Downloading from {} geotags...'.format(len(geo_urls)))

    try:
        print('Starting {} workers...'.format(cpu_count()))
        workers = Pool(processes=cpu_count())
        results = workers.imap_unordered(downloadPage, geo_urls);

        with open(outputFile, 'w+') as o:
            fieldnames = ['x', 'y', 'text']
            writer = csv.DictWriter(o, delimiter=',', fieldnames=fieldnames)
            writer.writeheader()

            for res in results:
                for x, y, t in res:
                    # write text back into csv with fake geotags
                    if (t != '' and x is not None and y is not None and t is not None):
                        writer.writerow({
                            'x': x,
                            'y': y,
                            'text': t
                        })
                    else:
                        print("Skipping (empty text field)")
    except KeyboardInterrupt:
        print("\nStopped by user")
        sys.exit(1)


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='the input file with the URLs', required=True)
    parser.add_argument('--output', type=str, help='the output csv file', required=True)
    args = parser.parse_args()
    # get abs path from file
    ifilePath = os.path.abspath(args.input)
    ofilePath = os.path.abspath(args.output) 
    main(ifilePath, ofilePath)