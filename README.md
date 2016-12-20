# topic-zoomer
This is the project for the Big Data course at the University of Trento.

The goal is to categorise geolocalised URLs. It is possible to select an area of
interest by means of a top left and a bottom right point and a step S to further
divide this area into squares (of size SxS).

# Requirements
* python3
* pip dependencies installed via `pip install -r requirements.txt`
* working spark environment

# How to run the tool
## Example
```bash
cd /path/to/repo/src
/path/to/spark-submit topic_zoomer.py --tlx 0 --tly 100 --brx 100 --bry 0 --step 50 --dataset ../data/test.csv
```
