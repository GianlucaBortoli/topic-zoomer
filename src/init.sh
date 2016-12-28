#!/bin/bash

if [ "$EUID" -ne 0 ]; then
  echo "Please run it as root"
  exit 1
fi

if [ $# -eq 0 ]; then
  echo "No pip requirements file provided"
  exit 1
fi

REQS=$1 # the requirements.txt file

sudo apt-get install python3-pip
sudo pip3 install -r "$REQS"
