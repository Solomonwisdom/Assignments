#!/bin/bash

echo ""

echo -e "\nbuild docker hadoop image\n"
docker build -f hadoop.dockerfile -t solomonfield/hadoop:2.7.7 .

echo ""