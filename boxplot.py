import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
  with open('results/random_forest/metrics.json', 'r') as file:
    rf_results = json.load(file)
  
  accuracies = rf_results['accuracy']
  fscores = rf_results['fscore']

  # x_labels = list(accuracies.keys())
  # y_values = list(accuracies.values())

  column_label_map = {
    'no_sample': 'No Sampling',
    'nearest_neighbor': 'Nearest Neighbor', 
    'undersampling': 'Undersampling',
    'naive_bayes': 'Naive Bayes'
  }

  df_accuracy = pd.DataFrame.from_dict(accuracies, orient='index').transpose().rename(column_label_map)
  df_fscore = pd.DataFrame.from_dict(fscores, orient='index').transpose().rename(column_label_map)

  plt.figure(figsize=(8, 6))
  sns.boxplot(data=df_accuracy)
  plt.xlabel('Stratgies')
  plt.ylabel('Accuracy')
  plt.title('Accuracy vs Strategy')
  plt.show()

  plt.savefig('rf_accuracy_boxplot.png')
  plt.show()



if __name__ == '__main__':
  main()