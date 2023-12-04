import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import train_test_split
import sys
import argparse

class SamplingNB:
  def __init__(self):
    pass

  def generate(self, features, labels, target_label, points_wanted, guassian_cols, bernoulli_cols):
    self.points_wanted = points_wanted
    self.target_label = target_label

    target_name = labels.name
    targets = labels.to_numpy()
  
    g_features = features.iloc[:, guassian_cols] 
    g_points = self.__gaussian(features=g_features, targets=targets)
    g_df = pd.DataFrame(g_points, columns=g_features.columns)

    b_features = features.iloc[:, bernoulli_cols]
    b_points = self.__bernoulli(b_features, targets)
    b_df = pd.DataFrame(b_points, columns=b_features.columns)

    new_labels = np.array([target_label for i in range(points_wanted)])
    new_points = pd.concat([g_df, b_df], axis=1)
    new_points[target_name] = new_labels

    return new_points

  def __gaussian(self, features, targets):
    X_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(x_test)

    # Parameters for the Gaussian distribution
    n_dimensions = len(features.columns)  # Specify the number of dimensions for the points
    num_points = 10000   # Specify the number of points to generate

    # Label of point we wish to generate
    label_of_interest = self.target_label

    # Define mean and standard deviation for each dimension
    means = features.mean()         # Mean for each dimension
    std_deviations = features.std()  # Standard deviation for each dimension 
    points_wanted = self.points_wanted
    points_added = 0

    new_points = []
    while points_added <= points_wanted:
      # Generate n-dimensional points following a Gaussian distribution
      generated_points = np.random.normal(means, std_deviations, (num_points, n_dimensions))
      generated_points_df = pd.DataFrame(generated_points, columns=features.columns)

      y_pred = clf.predict(generated_points_df)

      points_of_interest = generated_points[y_pred == label_of_interest]
      new_points.extend(points_of_interest)
      points_added += len(points_of_interest)

    final_points = new_points[:points_wanted]
    return final_points

  def __bernoulli(self, features, targets):
    X_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)

    clf = BernoulliNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(x_test)

    # Parameters for the Bernoulli distribution
    n_dimensions = len(features.columns)  # Specify the number of dimensions for the points
    num_points = 10000   # Specify the number of points to generate

    # label of minimum class
    label_of_interest = self.target_label

    # Define mean and standard deviation for each dimension
    means = features.mean()         # Mean for each dimension
    
    points_wanted = self.points_wanted
    points_added = 0
    # store the new points we want
    new_points = []
    while points_added <= points_wanted:
      # Generate n-dimensional points following a Gaussian distribution
      # Not 100% sold that p=means here, but the mean of a bernoulli distribution is p so it's prob fine
      generated_points = np.random.binomial(n=1, p=means, size=(num_points, n_dimensions))
      generated_points_df = pd.DataFrame(generated_points, columns=features.columns)

      y_pred = clf.predict(generated_points_df)

      points_of_interest = generated_points[y_pred == label_of_interest]
      new_points.extend(points_of_interest.astype(int))
      points_added += len(points_of_interest)
  
    final_points = new_points[:points_wanted]
    return final_points


# example usage -- oversampling
def main():
  path_to_input = sys.argv[1]
  flag_args = sys.argv[2:]
  parser = argparse.ArgumentParser()
  parser.add_argument('-l', '--label', help='Column name of the label')
  parser.add_argument('-m', '--minority', type=int, help='Label value of the minority class')
  parser.add_argument('-M', '--majority', type=int, help='Label value of the majority class')
  parser.add_argument('-o', '--output', help='File name for the ouput class')
  args = parser.parse_args(flag_args)

  df = pd.read_csv(path_to_input)
  features = df.iloc[:, :-1]
  labels = df.iloc[:, -1]
  points = len(df[df[args.label] == args.minority])

  nb_sampler = SamplingNB()

  new_minority = nb_sampler.generate(features, labels, target_label=args.minority, points_wanted=points, 
                                   guassian_cols=slice(0, 10), bernoulli_cols=slice(10, 28))

  majority = df[df[args.label] == args.majority]
  balanced_data = pd.concat([majority, new_minority])
  
  pd.DataFrame.to_csv(balanced_data, args.output, index=False)
  

if __name__ == '__main__':
  main()