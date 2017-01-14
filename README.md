# topic-zoomer
This is the project for the Big Data course at the University of Trento.

The goal is to categorise geolocalised URLs. It is possible to select an area of
interest by means of a top left and a bottom right point and a step S to further
divide this area into squares (of size SxS).

# Requirements
* python3 version<=3.5 (it will not work on python3.6 due to some incompatibility between python3.6 and pySpark)
* pip dependencies installed via `pip install -r requirements.txt` (see _init.sh_ script)
* working Spark environment

# How to run the tool
## Example to generate the dataset from URL file
```bash
cd /path/to/repo/data
python crawler.py --input 100_wikipedia_urls --output out.csv --min 0 --max 100
```

## Example to run the topic extractor on Spark
```bash
cd /path/to/repo/src
/path/to/spark-submit topic_zoomer.py 5 0 100 100 0 100 ../data/test.csv
```
