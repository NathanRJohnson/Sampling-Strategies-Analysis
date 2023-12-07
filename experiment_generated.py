from sklearn.metrics import precision_score as precision
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from os.path import join
import pandas as pd
import numpy as np
import json
import os

def main():
    #metrics we will be using for random forest and svc
    metrics = ['precision', 'recall', 'fscore', 'confusion_matrix']
    
    #creating dictionaries to hold results
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

    #loops through each strategy
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
        
        # tries each strategy on each data file. 
        # Since the generated noise data has different variations (different ratios of majority to minority class, different levels of noise)
        # we use each strategy on each
        for fname in os.listdir(generated_path):
            if fname.endswith('.csv'):
                print(f'{strategy}: {fname}')
                
                # splitting up the file name to get ratio and noise level
                fname_plain = fname.split('.csv')[0]
                fname_plain = fname_plain.replace('_', '-')
                fname_sections = fname_plain.split('-')
                
                #if this file is the first one using this ratio we have to initialize it as an empty dict
                if (fname_sections[0] not in rf_precision[strategy].keys()):
                    rf_precision[strategy][fname_sections[0]] = {}
                    rf_recall[strategy][fname_sections[0]] = {}
                    rf_f_scores[strategy][fname_sections[0]] = {}
                    svc_precision[strategy][fname_sections[0]] = {}
                    svc_recall[strategy][fname_sections[0]] = {}
                    svc_f_scores[strategy][fname_sections[0]]= {}
                
                # initialize strategy, ratio, and noise level combo in each dictionary
                rf_precision[strategy][fname_sections[0]][fname_sections[1]] = []
                rf_recall[strategy][fname_sections[0]][fname_sections[1]] = []
                rf_f_scores[strategy][fname_sections[0]][fname_sections[1]] = []

                svc_precision[strategy][fname_sections[0]][fname_sections[1]] = []
                svc_recall[strategy][fname_sections[0]][fname_sections[1]] = []
                svc_f_scores[strategy][fname_sections[0]][fname_sections[1]] = []
                
                #initialize classifiers
                random_forest = RandomForestClassifier(n_estimators=500)
                svc = LinearSVC(dual='auto')

                # get training data
                if strategy == 'no_sample':
                    train_data = pd.read_csv(join(test_path, f'train_{fname}'))
                else:
                    train_data = pd.read_csv(join(train_path, strategy, f'balanced_train_{fname}'))
                X_train = train_data.drop('label', axis=1).to_numpy()
                y = train_data['label'].to_list()
                
                #fit the classifiers
                random_forest.fit(X_train, y)
                svc.fit(X_train, y)

                #get testing data
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

                #store results
                rf_precision[strategy][fname_sections[0]][fname_sections[1]].append(precision(y_pred=rf_pred, y_true=true_labels, pos_label=1))
                rf_recall[strategy][fname_sections[0]][fname_sections[1]].append(recall(y_pred=rf_pred, y_true=true_labels, pos_label=1))
                rf_f_scores[strategy][fname_sections[0]][fname_sections[1]].append(f1(y_pred=rf_pred, y_true=true_labels, pos_label=1))

                svc_precision[strategy][fname_sections[0]][fname_sections[1]].append(precision(y_pred=svc_pred, y_true=true_labels, pos_label=1))
                svc_recall[strategy][fname_sections[0]][fname_sections[1]].append(recall(y_pred=svc_pred, y_true=true_labels, pos_label=1))
                svc_f_scores[strategy][fname_sections[0]][fname_sections[1]].append(f1(y_pred=svc_pred, y_true=true_labels, pos_label=1))
        
        # create confusion matrix for both classifiers    
        rf_cm = confusion_matrix(y_pred=all_rf_predictions, y_true=all_true_labels, labels=[0, 1])
        rf_confusion_matricies[strategy] = rf_cm

        rfcmd = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=['Majority', 'Minority'])
        rfcmd.plot().figure_.savefig(join(rf_results_path, f'{strategy}_confusion_matrix.png'))

        svc_cm = confusion_matrix(y_pred=all_svc_predictions, y_true=all_true_labels, labels=[0, 1])
        svc_confusion_matricies[strategy] = svc_cm

        svccmd = ConfusionMatrixDisplay(confusion_matrix=svc_cm, display_labels=['Majority', 'Minority'])
        svccmd.plot().figure_.savefig(join(svc_results_path, f'{strategy}_confusion_matrix.png'))

    # save metrics
    with open(join(rf_results_path, 'metrics.json'), 'w') as file:
        json.dump({'precision': rf_precision, 'recall': rf_recall, 'fscore': rf_f_scores}, file)

    with open(join(svc_results_path, 'metrics.json'), 'w') as file:
        json.dump({'precision': svc_precision, 'recall': svc_recall, 'fscore': svc_f_scores}, file)


if __name__ == '__main__':
    main()