import json
import pandas as pd

def main():
  with open('results/random_forest/metrics.json', 'r') as file:
    rf_results = json.load(file)

  print('---Random Forest ---\n', make_table(rf_results), '\n\n')

  with open('results/svc/metrics.json', 'r') as file:
    svc_results = json.load(file)
  
  print('---Support Vector Machine ---\n', make_table(svc_results))


def make_table(data):
  column_label_map = {
    'no_sample': 'No Sampling',
    'nearest_neighbor': 'Clustering', 
    'undersampling': 'Undersampling',
    'naive_bayes': 'Naive Bayes',
    'oversampling': 'SMOTE',
    'gan': 'GAN'
  }
  table = pd.DataFrame()
  metrics = ['precision', 'recall', 'fscore']
  for metric in metrics:
    metric_means = pd.DataFrame.from_dict(data[metric], orient='index').transpose().rename(columns=column_label_map).mean()
    table[metric] = metric_means
  
  # delta column
  base_fscore = table['fscore']['No Sampling']
  scores = [0]
  for i in range(1,6):
    scores.append(table['fscore'].iloc[i] - base_fscore)
  table['Delta F-Score'] = scores
  
  return table.rename(columns={'precision':'Precision', 'recall':'Recall', 'fscore': 'F-Score'}).transpose()

if __name__ == '__main__':
  main()