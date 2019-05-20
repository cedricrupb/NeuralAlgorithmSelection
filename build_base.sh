#!/bin/bash

zipfile=pesco.zip

if [ ! -f $zipfile ] ; then
  ./scripts/setup-pesco.sh
fi

docker build -t $BASE_REPOSITORY -f ./docker-nas-base .

echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
docker push $BASE_REPOSITORY:latest
