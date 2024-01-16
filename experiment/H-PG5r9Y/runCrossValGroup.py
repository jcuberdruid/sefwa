import sys
import json
from cl.userExtension.train_and_classify import Classifier
import numpy as np
import csv
import argparse
from cl.ut.utils import config

path = "experiment/H-PG5r9Y/kfolder"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Program to process and classify data kfold.')

    parser.add_argument('-subject', dest='subject', action='store',help='current subject ')
    parser.add_argument('-kfold',dest='kfold', action='store',help='which kfold is test set')

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    if not args.subject:
        exit(1) 
    if not args.kfold:
        exit(1) 

    subject = args.subject
    kfold = args.kfold

    train_data = np.load(path+"train_data.npy")
    train_labels = np.load(path+"train_labels.npy")
    test_data = np.load(path+"test_data.npy")
    test_labels = np.load(path+"test_labels.npy")
    defaults = config.readConfig()

    instance_classify = Classifier("test_kfold", subject, 200, 500, defaults['model'], defaults['dataset'])
    accuracy = instance_classify.classify(train_data, train_labels, test_data, test_labels)

    with open('kfoldScores.csv', 'a') as f:
        row = f"{subject},{kfold},{accuracy}\n"
        f.write(row)

main()
