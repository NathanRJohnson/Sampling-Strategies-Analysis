import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def make_plots(precision, recall, fscores, clf):
  # plt.figure(figsize=(8, 6))
  # sns.boxplot(data=accuracies)
  # plt.xlabel('Stratgy')
  # plt.ylabel('Accuracy')
  # plt.title(f'Accuracy by Strategy - {clf}')

  # plt.savefig(f'{clf}_accuracy_boxplot.png')

  plt.figure(figsize=(8, 6))
  sns.boxplot(data=precision)
  plt.xlabel('Stratgy')
  plt.ylabel('Precision')
  plt.title(f'Precision by Strategy - {clf}')

  plt.savefig(f'{clf}_precision_boxplot.png')

  plt.clf()

  plt.figure(figsize=(8, 6))
  sns.boxplot(data=recall)
  plt.xlabel('Stratgy')
  plt.ylabel('Recall')
  plt.title(f'Recall by Strategy - {clf}')

  plt.savefig(f'{clf}_recall_boxplot.png')

  plt.clf()
  
  plt.figure(figsize=(8, 6))
  sns.boxplot(data=fscores)
  plt.xlabel('Stratgy')
  plt.ylabel('F Score')
  plt.title(f'F Score by Strategy - {clf}')

  plt.savefig(f'{clf}_fscore_boxplot.png')

def main():
  with open('results/random_forest/metrics.json', 'r') as file:
    rf_results = json.load(file)
  
  # rf_accuracies_list = rf_results['accuracy']
  rf_fscores_list = rf_results['fscore']
  rf_precisions_list = rf_results['precision']
  rf_recalls_list = rf_results['recall']


  with open('results/svc/metrics.json', 'r') as file:
    svc_results = json.load(file)
  
  # svc_accuracies_list = svc_results['accuracy']
  svc_fscores_list = svc_results['fscore']
  svc_precisions_list = svc_results['precision']
  svc_recalls_list = svc_results['recall']


  column_label_map = {
    'no_sample': 'No Sampling',
    'nearest_neighbor': 'Clustering', 
    'undersampling': 'Undersampling',
    'naive_bayes_95': 'Naive Bayes',
    'oversampling': 'SMOTE',
    'gan': 'GAN'
  }

  # rf_accuracy = pd.DataFrame.from_dict(rf_accuracies_list, orient='index').transpose().rename(columns=column_label_map)
  rf_precision = pd.DataFrame.from_dict(rf_precisions_list, orient='index').transpose().rename(columns=column_label_map)
  rf_recall = pd.DataFrame.from_dict(rf_recalls_list, orient='index').transpose().rename(columns=column_label_map)
  rf_fscore = pd.DataFrame.from_dict(rf_fscores_list, orient='index').transpose().rename(columns=column_label_map)

  make_plots(rf_precision, rf_recall, rf_fscore, "Random Forest")

  # svc_accuracy = pd.DataFrame.from_dict(svc_accuracies_list, orient='index').transpose().rename(columns=column_label_map)
  svc_fscore = pd.DataFrame.from_dict(svc_fscores_list, orient='index').transpose().rename(columns=column_label_map)
  svc_precision = pd.DataFrame.from_dict(svc_precisions_list, orient='index').transpose().rename(columns=column_label_map)
  svc_recall = pd.DataFrame.from_dict(svc_recalls_list, orient='index').transpose().rename(columns=column_label_map)

  make_plots(svc_precision, svc_recall, svc_fscore, "Support Vector Machine")

if __name__ == '__main__':
  main()