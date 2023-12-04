import numpy as np 
import pandas as pd
import random
from random import randrange
from sklearn.neighbors import NearestNeighbors
import sys
import argparse

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

def nn(X_data, K):
    neighbours = NearestNeighbors(n_neighbors=K+1, metric='euclidean',algorithm='kd_tree').fit(X_data)
    euclidean,indices= neighbours.kneighbors(X_data)
    return euclidean, indices

def SMOTE(X_data, oversampling_needed, K, majority_len):
    print(oversampling_needed)
    shape = X_data.shape
    euclidean, indices = nn(X_data, K)
    fake_samples = np.empty((oversampling_needed*shape[0], shape[1]))
    idx = 0
    for i in range(shape[0]):
        for j in range(oversampling_needed):
            neigh = randrange(1, K+1)
            #print("here",  X_data.iloc[indices[i, neigh]])
            diff = X_data.iloc[indices[i, neigh]] - X_data.iloc[i]
            gap = random.uniform(0,1)
            fake_samples[idx] = X_data.iloc[i] + gap*diff
            idx+=1
    samples_df = pd.DataFrame(fake_samples, columns =X_data.columns)
    samples_df = samples_df[:(majority_len-len(X_data))] #removing the excess samples made
    return samples_df
            
        


def main():
    path_to_input, inputs=parse_args()
    print(inputs )
    #Creating the dataframe to hold the data from covtype.data
    WA_columns = ['wilderness_A_' + str(x + 1) for x in range(4)]
    soil_columns = ['soil_T_' + str(x + 1) for x in range(40)]

    columns = ["elevation", "aspect", "slope", "hydro_horizontal_dist", "hydro_vertical_dist", "road_horizontal_dist" "shade_9am", "shade_noon", "shade_3pm", "firepoints_horizontal_dist"] + WA_columns + soil_columns + ["cover_type"]

    whole_dataset = pd.read_csv(path_to_input, header=None, names=columns)

    subset_data = pd.concat([whole_dataset[whole_dataset[inputs.label] == inputs.majority], whole_dataset[whole_dataset[inputs.label] == inputs.minority]]).reset_index().drop(columns=['index'])
    minority_class_data = whole_dataset[whole_dataset[inputs.label] == inputs.minority].reset_index().drop(columns=['index'])

    majority_len = len(subset_data[subset_data[inputs.label] == inputs.majority])
    minority_len = len(subset_data[subset_data[inputs.label] == inputs.minority])
    print("# of samples with ",inputs.label, " ", inputs.minority, ": ", minority_len)
    print("# of samples with ", inputs.label, " ", inputs.majority, ": ", majority_len)

    #subset_labels = subset_data[inputs.label]
    #minority_class_labels = minority_class_data[inputs.label]
    

    samples_needed = int(majority_len/minority_len)#ex. 12 means 12 times # of minority class
    neighbours = 5 #number of neighbours to use
    sampled_instances = SMOTE(minority_class_data, samples_needed, neighbours, majority_len)
    balanced_set =pd.concat([subset_data, sampled_instances]).reset_index().drop(columns=['index'])
    if (inputs.label == 'cover_type'):
        balanced_set['cover_type'] = balanced_set['cover_type'].astype(int)

    print(balanced_set['cover_type'].value_counts())
    balanced_set.to_csv(inputs.output)
    
if __name__ == '__main__':
    main()