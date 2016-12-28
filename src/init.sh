#!/bin/bash

if [ $# -eq 0 ]; then
  echo "No pip requirements file provided"
  exit 1
fi

REQS=$1 # the requirements.txt file

sudo apt-get install -y python3-pip
sudo pip3 install -r "$REQS"

# set python3 as default
sudo rm /usr/bin/python
sudo ln -s /usr/bin/python3 /usr/bin/python