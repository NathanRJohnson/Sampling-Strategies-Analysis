import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as col

''' 
Creates a cluster using a mean and a covariance matrix.
Ideally you can specify the dimensions (maybe the covariance matrix has this info).
:param mean: The mean of the distribution.
:param matrix: A covariance matrix for describing the distribution.
:param size: The number of points to generate.
:returns: a Dataframe containing 'size' points drawn from a normal distribution.
'''
def create_cluster(mean, matrix, size):
  # create random ndarray
  ndarray = np.random.multivariate_normal(mean, matrix, size)
  return pd.DataFrame(ndarray)

'''
Combines clusters into a single dataframe.

:param a: first cluster.
:param b: second cluster.
:returns: The combined clusters in a dataframe.
'''
def dataset_from_clusters(a, b):
  frames = [a, b]
  return pd.concat(frames)

'''
Plots the cluster in 2D.

:param dataframe: A dataframe representing a cluster.
:param labels: The possible labels for a point in the cluster.
:param name: A save location for the plot.
'''
def plot_data(dataframe, labels, name):
  colors = ['red', 'green', 'orange', 'blue']
  scatter = plt.scatter(dataframe[0], dataframe[1], c=dataframe['label'], cmap=col.ListedColormap(colors))
  plt.xlabel('Alpha')
  plt.ylabel('Beta')
  plt.legend(handles=scatter.legend_elements()[0], labels=labels, loc='upper left')
  plt.savefig(name)
  plt.clf()

'''
Plots the cluster in 3D.

:param dataframe: A dataframe representing a cluster.
:param labels: The possible labels for a point in the cluster.
:param name: A save location for the plot.
'''
def plot_data_3d(dataframe, labels, name):
  colors = ['red', 'green', 'orange', 'blue']
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  scatter = ax.scatter(dataframe[0], dataframe[1], dataframe[2],  c=dataframe['label'], cmap=col.ListedColormap(colors), alpha=0.1)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.legend(handles=scatter.legend_elements()[0], labels=labels, loc='upper left')
  plt.savefig(name)
  plt.clf()

'''
Adds a label to the cluster.

:param df: A dataframe representing a cluster.
:param label: The value of the label to be added.
'''
def add_label(df, label):
  label_vector = np.full((len(df), 1), label)
  df['label'] = label_vector
  
'''
Translates the cluster along an axis.

:param dataframe: A dataframe representing a cluster.
:param label: The value of the label to be added.
'''
def shift_cluster(df, column, amount):
  shift_vector = np.full((len(df), 1), amount)
  df[column] += shift_vector.flatten()


def label_probability_matrix(df, labels):
  ## I get the count for each label
  num_labels = len(labels)
  if num_labels < 2:
    return
  matrix = [[] for i in range(num_labels)]
  counts = [0 for i in range(num_labels)]
  N = 0
  for label in df['label']:
    counts[label] += 1
    N += 1
  for i in range(num_labels):
    for j in range(num_labels):
      if j != i:
        matrix[i].append(counts[j] / (N - counts[i]))

  return matrix

## function to generate a random number, and pick a label
def get_new_label(label_probability_matrix, old_label):
  row = label_probability_matrix[old_label]
  labels = [i for i in range(len(label_probability_matrix)) if i != old_label]
  rand = np.random.random(1)[0]
  for i in range(len(row) - 1):
    # row[i] is the threshold
    if rand < row[i]:
      return labels[i]
  # if none of the not last ones, return the last
  return labels[-1]


## A function to flip labels
def add_label_noise(df, labels, noise_ratio):
  if noise_ratio <= 0:
    return df
  
  noise_df = df.copy(deep=True)
  ## Make it work for even splits rn
  lpm = label_probability_matrix(df, labels)
  num_clusters = len(labels)
  ## TODO: add logic to compute the new_label
  for i in range(len(noise_df['label'])):
    rand = np.random.random(1)[0]
    if rand < noise_ratio:
      new_label = get_new_label(lpm, int(noise_df.iloc[i]['label']))
      noise_df.iloc[i, noise_df.columns.get_loc('label')] = new_label
  
  return noise_df


def main():

  mean = [0, 0, 0]
  matrix = [[1, 6, 16],[8, 3, 17],[19, 7, 11]]
  labels=[0, 1]
  save_path = 'generated/'
  image_save_path = save_path + 'plots/'

  total_points = 10000
  proportoin_levels = [0.7, 0.8, 0.9]
  noise_levels = [0, 0.05, 0.1, 0.2]

  for proportion in proportoin_levels:

    clusterA = create_cluster(mean, matrix, int(proportion*total_points))
    add_label(clusterA, labels[0])
    shift_cluster(clusterA, 0, -6)
    shift_cluster(clusterA, 1, -7)

    clusterB = create_cluster(mean, matrix, int((1-proportion)*total_points))
    add_label(clusterB, labels[1])
    shift_cluster(clusterB, 0, 7)
    shift_cluster(clusterA, 2, 5)

    data = dataset_from_clusters(clusterA, clusterB)

    for noise in noise_levels:

      preturbed_df = add_label_noise(data, labels, noise)
      title = "{}:{}-{}_data".format(int(100*proportion), int(100*(1-proportion)), int(100*noise))
      pd.DataFrame.to_csv(preturbed_df, save_path+title+'.csv', index=False)
      plot_data_3d(preturbed_df, labels, image_save_path+title)

  # plot_data(data, "30_noise")

if __name__ == '__main__':
  main()