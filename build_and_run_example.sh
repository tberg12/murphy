#!/bin/sh

# Build
echo "Building source"
find . | grep ".java$" > sources.txt
mkdir example/bin
javac -d example/bin -classpath lib/jblas-1.2.3.jar:lib/JCuda-All-0.6.5-bin-apple-x86_64/jcublas-0.6.5.jar:lib/JCuda-All-0.6.5-bin-apple-x86_64/jcuda-0.6.5.jar:lib/JCuda-All-0.6.5-bin-apple-x86_64/jcufft-0.6.5.jar:lib/JCuda-All-0.6.5-bin-apple-x86_64/jcurand-0.6.5.jar:lib/JCuda-All-0.6.5-bin-apple-x86_64/jcusparse-0.6.5.jar:lib/lbfgsb_wrapper-1.1.2/lbfgsb_wrapper-1.1.2.jar:lib/liblinear-1.94.jar @sources.txt
rm sources.txt
echo

# Run
echo "Running example"
java -classpath lib/jblas-1.2.3.jar:lib/JCuda-All-0.6.5-bin-apple-x86_64/jcublas-0.6.5.jar:lib/JCuda-All-0.6.5-bin-apple-x86_64/jcuda-0.6.5.jar:lib/JCuda-All-0.6.5-bin-apple-x86_64/jcufft-0.6.5.jar:lib/JCuda-All-0.6.5-bin-apple-x86_64/jcurand-0.6.5.jar:lib/JCuda-All-0.6.5-bin-apple-x86_64/jcusparse-0.6.5.jar:lib/lbfgsb_wrapper-1.1.2/lbfgsb_wrapper-1.1.2.jar:lib/liblinear-1.94.jar:.:example/bin example.ExampleSystem
echo

