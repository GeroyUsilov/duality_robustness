#!/bin/bash
sample=$1
echo "$sample processing"



python controller_ev.py $sample

echo "$sample is done"
