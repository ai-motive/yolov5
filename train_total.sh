#!/bin/bash

cd yolo_learn
nohup python yolo_learn.py --op_mode TRAIN_TEST --ini_fname yolo_learn_total.ini > $(date +%y%m%d)_train_total.txt &
cd ..