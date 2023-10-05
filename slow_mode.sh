#!/bin/bash
sample=$1
echo "$sample processing"



python slow_mode_ev.py $sample

echo "$sample is done"
