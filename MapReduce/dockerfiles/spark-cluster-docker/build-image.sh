#!/bin/bash

echo ""

echo -e "\nbuild spark image\n"
docker build -f spark.dockerfile -t solomonfield/spark:2.3.3 .

echo ""