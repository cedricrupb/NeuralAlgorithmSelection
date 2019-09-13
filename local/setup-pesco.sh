#!/bin/bash

gitrepro=https://github.com/cedricrupb/cpachecker.git
gitdir=cpachecker/
targetdir=pesco/
target=pesco.zip
result=$?

echo "Checking if PeSCo exists in dir."

if [ ! -d "$gitdir" ]; then
  echo "Checkout PeSCo"
  git clone $gitrepro

  if [ $result -ne 0 ]; then
    exit $result
  fi

  cd $gitdir
else
  echo "Pull from master"
  cd $gitdir
  git pull

  if [ $result -ne 0 ]; then
    exit $result
  fi

fi

if [ ! -f "cpachecker.jar" ] && [ ! -d "../$targetdir" ]; then
  echo "Building PeSCo"
  ant jar

  if [ $result -ne 0 ]; then
    exit $result
  fi

fi

if [ -f "cpachecker.jar" ]; then
  echo "Reduce to necessary files"

  if [ -d "../$targetdir" ]; then
    rm -r "../$targetdir"
  fi
  mkdir "../$targetdir"

  mv cpachecker.jar "../$targetdir/cpachecker.jar"
  mv resources/ "../$targetdir/resources/"
  mv scripts/ "../$targetdir/scripts/"
  mv config/ "../$targetdir/config/"
  mv lib/ "../$targetdir/lib/"
  mv doc/ "../$targetdir/doc/"
  mv "License_Apache-2.0.txt" "../$targetdir/"
  mv "Copyright.txt" "../$targetdir/"
  mv "README.md" "../$targetdir/"
  mv "INSTALL.md" "../$targetdir/"
fi

echo "Cleanup resources"
cd ".."

if [ -d $gitdir ]; then
  rm -rf $gitdir
fi

if [ -d $targetdir ] && [ ! -f $target ]; then
  zip -r $target $targetdir
  rm -rf $targetdir
fi
echo "Success."
