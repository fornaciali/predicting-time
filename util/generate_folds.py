import os
import argparse
import numpy as np

from sklearn.model_selection import KFold


def get_args():
    parser = argparse.ArgumentParser(description='Generates and persists the 5-folds.')
    parser.add_argument('folder_name', type=str)
    args = parser.parse_args()
    return args


def save_fold(folder, fold_ID, train, test):
    '''
    Save the folds into TXT files, using 2 lines: the first to the training data 
    and the second for the testing data. Data is the ID of the days, from 1 to 50. 

    Arguments:
    - folder (str) : directory to store the TXT files
    - fold_ID (int): the ID of the fold being saved 
    - train (list) : the traning set of the fold 
    - test (list) : the testing set of the fold
    '''

    file_name = os.path.join(folder, "fold_"+str(fold_ID)+".txt")
    f = open(file_name, "w")
    f.write("train=")
    f.write(','.join(map(str, train)))
    f.write("\n")
    f.write("test=")
    f.write(','.join(map(str, test)))
    f.write("\n")
    f.close()


def main():
    args = get_args()

    days = np.array(range(1, 51))

    kf = KFold(n_splits=5, shuffle=True)
    fold_ID = 1
    for train_index, test_index in kf.split(days):
        save_fold(args.folder_name, fold_ID, days[train_index], days[test_index])
        fold_ID += 1

    print("Fold files generated at: [{}]".format(args.folder_name))


if __name__ == '__main__':
    main() 