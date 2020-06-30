#!/bin/bash
. config.properties
mvn compile
mvn exec:java -Dexec.mainClass=RunModel -Dexec.cleanupDaemonThreads=false
if [[ "$inputFolder" == "" ]]; then
  inputFolder="Data/$data_name"
fi
if [[ "$outputFolder" == "" ]]; then
  outputFolder="$inputFolder/Output"
fi
python3 correctOutput.py -i "$outputFolder"/patternOutput."$noOfLines"_"$minimumSupport".txt -d "$inputFolder"/dict.json