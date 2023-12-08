from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

import sys
import argparse

def main():
  path_to_input = sys.argv[1]
  flag_args = sys.argv[2:]
  parser = argparse.ArgumentParser()
  parser.add_argument('-l', '--label', help='Column name of the label')
  parser.add_argument('-m', '--minority', type=int, help='Label value of the minority class')
  parser.add_argument('-M', '--majority', type=int, help='Label value of the majority class')
  parser.add_argument('-o', '--output', help='File name for the ouput class')
  args = parser.parse_args(flag_args)

  # load data
  df = pd.read_csv(path_to_input)

  # split data in to maj min 
  majority = df[df[args.label] == args.majority] #3
  minority = df[df[args.label] == args.minority] #4
  # print(len(majority), len(minority))
  
  # undersample maj
  kmean_model = KMeans(n_clusters=len(minority), n_init='auto')
  
  # # find the len(min) means
  means = kmean_model.fit(majority.to_numpy()).cluster_centers_
  # print(means)
  
  # # find the nearest neighbor for each of the means
  nn = NearestNeighbors(n_neighbors=1)
  nn.fit(majority.to_numpy())
  sampled_indicies = nn.kneighbors(means, return_distance=False).flatten()
  # print(sampled_indicies)
  
  # return as the undersampled maj
  sampled_maj = majority.iloc[sampled_indicies]
  # print(sampled_maj)

  # combine data
  new_data = pd.concat([sampled_maj, minority]).reset_index(drop=True)
  
  # return data set
  pd.DataFrame.to_csv(new_data, args.output, index=False) # '../data/sampled/knn.csv'

if __name__ == '__main__':
  main()