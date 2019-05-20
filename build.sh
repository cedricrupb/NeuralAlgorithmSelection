#!/bin/bash

zipfile=pesco.zip

if [ ! -f $zipfile ] ; then
  ./scripts/setup-pesco.sh
fi

docker build -t cedricr:pesco .
