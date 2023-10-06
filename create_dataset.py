import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from plotread import *
import shutil


def modify_dataset(sequences, features):
    for sequence in sequences:
        for feature in features:
            getattr(sequence, feature)[:] = 0
    
    return sequences
                
def save_dataset(data_path, save_path, file_list, cols, modified_datax, features):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    src_files = os.listdir(data_path)
    for file_name in src_files:
        full_file_name = os.path.join(data_path, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, save_path)
        
    for i, file in enumerate(file_list):
        original_file_path = os.path.join(data_path, file)
        save_file_path = os.path.join(save_path, file)
        
        df = pd.read_csv(save_file_path, sep="\s+", names=cols)
        df = df.set_index('time')
        
        for j in features:
            df.loc[:, j] = modified_datax[i][j]
            
        df.reset_index(inplace=True)    
        df.insert(0, 'index', df.index)

        df.to_csv(save_file_path, sep=' ', header=False, index=False)

def main():
    ###########################################################################
    # Parser Definition
    ###########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-datapath', type=str, default="Dataset1/")
    parser.add_argument('-savepath', type=str, default="Dataset1x/")
    parser.add_argument('-extension', type=str, default=".dat")
    parser.add_argument('-features', nargs='+', type=str, help='Specify 1 to 4 features')
    opt = parser.parse_args()

    ###########################################################################
    # Variables Definition
    ###########################################################################
    nin = ['time', 'PLML2', 'PLMR', 'AVBL', 'AVBR']
    nout = ['time', 'DB1', 'LUAL', 'PVR', 'VB1']
    neurons = ['time','DB1','LUAL','PVR','VB1','PLML2','PLMR','AVBL','AVBR']

    ###########################################################################
    # Read data
    ###########################################################################
    files = getdata(opt.datapath, opt.extension)
    train, valid, test = splitdata(files)
    trainx, trainy = readdata(opt.datapath, train, neurons, nin, nout)
    validx, validy = readdata(opt.datapath, valid, neurons, nin, nout)
    testx, testy = readdata(opt.datapath, test, neurons, nin, nout)
    
    modified_testx = modify_dataset(testx, opt.features)
    
    save_dataset(opt.datapath, opt.savepath, test, neurons, modified_testx, opt.features)

        
        
if __name__ == "__main__":
    main()