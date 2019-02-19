#!/bin/bash
gsutil cp -r gs://uga-dsp/project2 .
cd ./data
for l in $(ls *tar)
do
    tar xvf $l
done
exit 0