#!/bin/bash
mvn compile
mvn exec:java -Dexec.mainClass=RunModel -Dexec.cleanupDaemonThreads=false