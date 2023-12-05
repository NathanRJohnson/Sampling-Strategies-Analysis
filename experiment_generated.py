from sklearn.metrics import precision_score as precision
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from os.path import join, exists
from os import makedirs
import pandas as pd
import numpy as np
import json
import os
# import seaborn as sns

def main():
    metrics = ['precision', 'recall', 'fscore', 'confusion_matrix']
    rf_precision = {}
    rf_recall = {}
    rf_f_scores = {}
    rf_confusion_matricies = {}

    svc_precision = {}
    svc_recall = {}
    svc_f_scores = {}
    svc_confusion_matricies = {}

    train_path = './data/sampled_generated/'
    test_path = './data/generated_split/'

    rf_results_path = f'results/random_forest_generated/'
    svc_results_path = f'results/svc_generated/'

    strategies = ['no_sample', 'oversampling', 'gan_generated', 'nearest_neighbor', 'undersampling'] #, 'naive_bayes']
    for strategy in strategies:
        rf_precision[strategy] = {}
        rf_recall[strategy] = {}
        rf_f_scores[strategy] = {}

        svc_precision[strategy] = {}
        svc_recall[strategy] = {}
        svc_f_scores[strategy] = {}

        all_rf_predictions = []
        all_svc_predictions = []

        all_true_labels = []

        generated_path = './data/generated'
        
        for fname in os.listdir(generated_path):
            if fname.endswith('.csv'):
                print(f'{strategy}: {fname}')
                fname_plain = fname.split('.csv')[0]
                fname_plain = fname_plain.replace('_', '-')
                fname_sections = fname_plain.split('-')
                if (fname_sections[0] not in rf_precision[strategy].keys()):
                    rf_precision[strategy][fname_sections[0]] = {}
                    rf_recall[strategy][fname_sections[0]] = {}
                    rf_f_scores[strategy][fname_sections[0]] = {}
                    svc_precision[strategy][fname_sections[0]] = {}
                    svc_recall[strategy][fname_sections[0]] = {}
                    svc_f_scores[strategy][fname_sections[0]]= {}
                    
                rf_precision[strategy][fname_sections[0]][fname_sections[1]] = []
                rf_recall[strategy][fname_sections[0]][fname_sections[1]] = []
                rf_f_scores[strategy][fname_sections[0]][fname_sections[1]] = []

                svc_precision[strategy][fname_sections[0]][fname_sections[1]] = []
                svc_recall[strategy][fname_sections[0]][fname_sections[1]] = []
                svc_f_scores[strategy][fname_sections[0]][fname_sections[1]] = []
                random_forest = RandomForestClassifier(n_estimators=500)
                svc = LinearSVC(dual='auto')

                # train
                if strategy == 'no_sample':
                    train_data = pd.read_csv(join(test_path, f'train_{fname}'))
                elif strategy == 'gan': # can be removed after resampling wit GAN
                    train_data = pd.read_csv(join(train_path, strategy, f'balanced_train_{fname}')).reset_index(drop=True).drop('Unnamed: 0', axis=1)
                else:
                    train_data = pd.read_csv(join(train_path, strategy, f'balanced_train_{fname}'))

                X_train = train_data.drop('label', axis=1).to_numpy()
                y = train_data['label'].to_list()
                random_forest.fit(X_train, y)
                svc.fit(X_train, y)

                test_data = pd.read_csv(join(test_path, f'test_{fname}'))
                true_labels = test_data['label'].to_list()
                all_true_labels.extend(true_labels)
            # The code snippet you provided is performing the testing phase of the machine learning models.

                # test
                X_test = test_data.drop('label', axis=1).to_numpy()
                rf_pred = random_forest.predict(X_test)
                all_rf_predictions.extend(rf_pred)

                svc_pred = svc.predict(X_test)
                all_svc_predictions.extend(svc_pred)

                
                rf_precision[strategy][fname_sections[0]][fname_sections[1]].append(precision(y_pred=rf_pred, y_true=true_labels, pos_label=1))
                rf_recall[strategy][fname_sections[0]][fname_sections[1]].append(recall(y_pred=rf_pred, y_true=true_labels, pos_label=1))
                rf_f_scores[strategy][fname_sections[0]][fname_sections[1]].append(f1(y_pred=rf_pred, y_true=true_labels, pos_label=1))

                svc_precision[strategy][fname_sections[0]][fname_sections[1]].append(precision(y_pred=svc_pred, y_true=true_labels, pos_label=1))
                svc_recall[strategy][fname_sections[0]][fname_sections[1]].append(recall(y_pred=svc_pred, y_true=true_labels, pos_label=1))
                svc_f_scores[strategy][fname_sections[0]][fname_sections[1]].append(f1(y_pred=svc_pred, y_true=true_labels, pos_label=1))
                
        #rf_cm = confusion_matrix(y_pred=all_rf_predictions, y_true=all_true_labels, labels=[0, 1])
        #rf_confusion_matricies[strategy] = rf_cm

        #rfcmd = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=['Majority', 'Minority'])
        #rfcmd.plot().figure_.savefig(join(rf_results_path, f'{strategy}_confusion_matrix.png'))

        #svc_cm = confusion_matrix(y_pred=all_svc_predictions, y_true=all_true_labels, labels=[3, 4])
        #svc_confusion_matricies[strategy] = svc_cm

        #svccmd = ConfusionMatrixDisplay(confusion_matrix=svc_cm, display_labels=['Majority', 'Minority'])
        #svccmd.plot().figure_.savefig(join(svc_results_path, f'{strategy}_confusion_matrix.png'))

    
    with open(join(rf_results_path, 'metrics.json'), 'w') as file:
        json.dump({'precision': rf_precision, 'recall': rf_recall, 'fscore': rf_f_scores}, file)

    with open(join(svc_results_path, 'metrics.json'), 'w') as file:
        json.dump({'precision': svc_precision, 'recall': svc_recall, 'fscore': svc_f_scores}, file)


if __name__ == '__main__':
    main()