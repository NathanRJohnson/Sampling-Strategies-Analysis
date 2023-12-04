import pandas as pd
import numpy as np

def main():
  covertype = pd.read_csv('../data/covertype.csv')

  # split the data into majority and minority
  majority_subset = covertype.loc[covertype['Cover_Type'] == 3]
  minority_subset = covertype.loc[covertype['Cover_Type'] == 4]

  # TODO: A ratio would be cool, so it' not always 50/50
  # TODO: might be worth looking for papers which discuss optimal undersampling ratios, 
  #       especially if this strategy underperforms relative to the others.
  # What happens if we have more samples from the minority than majority in the training data?
  num_points_to_remove = len(majority_subset) - len(minority_subset)

  # Generate random indices
  random_indices = np.random.choice(majority_subset.index, num_points_to_remove, replace=False)

  # Remove rows with random indices
  reduced_majority = majority_subset.drop(random_indices).reset_index().drop(columns=['index'])

  # stich the new majority to the old minority
  undersampled_df = pd.concat([minority_subset, reduced_majority], axis=0) 
  
  # save
  pd.DataFrame.to_csv(undersampled_df, '../data/sampled/undersampled.csv', index=False)

if __name__ == '__main__':
  main()
