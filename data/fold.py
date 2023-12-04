import pandas as pd
from sklearn.model_selection import KFold
## We want between participants
## Since it's time series data, we want to keep the data in order i.e not use the future to predict the past

def main():
  # for a specific feature
  # select a train size or test_size
  df = pd.read_csv('../data/covertype.csv')
  columns = list(df.columns)
  X = df.to_numpy()

  kfold = KFold(n_splits=5, shuffle=True)
  for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
    training_fold = X[train_idx]
    testing_fold = X[test_idx]

    df_train = pd.DataFrame(training_fold, columns=columns).reset_index(drop=True)
    df_test = pd.DataFrame(testing_fold, columns=columns).reset_index(drop=True)

    pd.DataFrame.to_csv(df_train, f'folded/train_fold_{fold}.csv', index=False)
    pd.DataFrame.to_csv(df_test, f'folded/test_fold_{fold}.csv', index=False)
    

if __name__ == '__main__':
  main()