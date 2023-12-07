import numpy as np 
import pandas as pd
import random
from random import randrange
from sklearn.neighbors import NearestNeighbors
import sys
import argparse

############
# performs SMOTE
############

#function to read command line arguments
def parse_args():
    path_to_input = sys.argv[1]
    flag_args = sys.argv[2:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', help='Column name of the label')
    parser.add_argument('-m', '--minority', type=int, help="label value of the minority class")
    parser.add_argument('-M', '--majority', type=int, help="label value of the majority class")
    parser.add_argument('-o', '--output', help='File name for the ouput class')
    args = parser.parse_args(flag_args)
    return path_to_input, args

#finds K nearest neighbours and returns the distances and indices
def nn(X_data, K):
    neighbours = NearestNeighbors(n_neighbors=K+1, metric='euclidean',algorithm='kd_tree').fit(X_data)
    euclidean,indices= neighbours.kneighbors(X_data)
    return euclidean, indices

# generates new samples.
# oversampling_needed parameter indicates how many samples we need
def SMOTE(X_data, oversampling_needed, K, majority_len):
    shape = X_data.shape
    euclidean, indices = nn(X_data, K)
    fake_samples = np.empty((oversampling_needed*shape[0], shape[1]))
    idx = 0
    for i in range(shape[0]):
        for j in range(oversampling_needed):
            neigh = randrange(1, K+1)
            diff = X_data.iloc[indices[i, neigh]] - X_data.iloc[i]
            gap = random.uniform(0,1)
            fake_samples[idx] = X_data.iloc[i] + gap*diff
            idx+=1
    samples_df = pd.DataFrame(fake_samples, columns =X_data.columns)
    samples_df = samples_df[:(majority_len-len(X_data))] #removing the excess samples made
    return samples_df
            
        


def main():
    path_to_input, inputs=parse_args()
    
    #getting data
    dataset = pd.read_csv(path_to_input)

    minority = dataset[dataset[inputs.label] == inputs.minority].reset_index(drop=True)
    majority = dataset[dataset[inputs.label] == inputs.majority].reset_index(drop=True)

    majority_len = len(majority)
    minority_len = len(minority)

    #set parameters
    samples_needed = int(majority_len/minority_len)#ex. 12 means 12 times # of minority class
    neighbours = 5 #number of neighbours to use
    
    #get generated data and concat to original dataset
    sampled_instances = SMOTE(minority, samples_needed, neighbours, majority_len)
    balanced_set = pd.concat([majority, sampled_instances]).reset_index(drop=True)
    
    if (inputs.label == 'Cover_Type'):
        balanced_set['Cover_Type'] = balanced_set['Cover_Type'].astype(int)

    #save data
    balanced_set.to_csv(inputs.output, index=False)
    
if __name__ == '__main__':
    main()