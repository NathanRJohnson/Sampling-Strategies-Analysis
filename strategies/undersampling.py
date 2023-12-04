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

  covertype = pd.read_csv(path_to_input)

  # split the data into majority and minority
  majority_subset = covertype.loc[covertype[args.label] == args.majority]
  minority_subset = covertype.loc[covertype[args.label] == args.minority]

  # TODO: A ratio would be cool, so it' not always 50/50
  # TODO: might be worth looking for papers which discuss optimal undersampling ratios, 
  #       especially if this strategy underperforms relative to the others.
  # What happens if we have more samples from the minority than majority in the training data?
  num_points_to_remove = len(majority_subset) - len(minority_subset)

  # Generate random indices
  random_indices = np.random.choice(majority_subset.index, num_points_to_remove, replace=False)

  # Remove rows with random indices
  reduced_majority = majority_subset.drop(random_indices).reset_index(drop=True)

  # stich the new majority to the old minority
  undersampled_df = pd.concat([minority_subset, reduced_majority], axis=0) 
  
  # save
  pd.DataFrame.to_csv(undersampled_df, args.output, index=False)

if __name__ == '__main__':
  main()
