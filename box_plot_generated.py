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

    plt.savefig(f'{clf}_precision_boxplot_generated.png')

    plt.clf()

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=recall)
    plt.xlabel('Stratgy')
    plt.ylabel('Recall')
    plt.title(f'Recall by Strategy - {clf}')

    plt.savefig(f'{clf}_recall_boxplot_generated.png')

    plt.clf()
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=fscores)
    plt.xlabel('Stratgy')
    plt.ylabel('F Score')
    plt.title(f'F Score by Strategy - {clf}')

    plt.savefig(f'{clf}_fscore_boxplot_generated.png')

def main():
    with open('results/random_forest_generated/metrics.json', 'r') as file:
        rf_results = json.load(file)
    
    # rf_accuracies_list = rf_results['accuracy']
    rf_fscores_list = {}
    rf_precisions_list = {}
    rf_recalls_list = {}
    for strat in rf_results['fscore']:
        rf_fscores_list[strat] = []
        rf_precisions_list[strat] = []
        rf_recalls_list[strat] = []
        for ratio in rf_results['fscore'][strat]:
            for noise in rf_results['fscore'][strat][ratio]:
                rf_fscores_list[strat].append(rf_results['fscore'][strat][ratio][noise][0])
                rf_precisions_list[strat].append(rf_results['precision'][strat][ratio][noise][0])
                rf_recalls_list[strat].append(rf_results['recall'][strat][ratio][noise][0])
    print(rf_fscores_list, "\n", rf_recalls_list)


    with open('results/svc_generated/metrics.json', 'r') as file:
        svc_results = json.load(file)
    
    # svc_accuracies_list = svc_results['accuracy']
    
    svc_fscores_list = {}
    svc_precisions_list = {}
    svc_recalls_list = {}
    for strat in svc_results['fscore']:
        svc_fscores_list[strat] = []
        svc_precisions_list[strat] = []
        svc_recalls_list[strat] = []
        for ratio in svc_results['fscore'][strat]:
            for noise in svc_results['fscore'][strat][ratio]:
                svc_fscores_list[strat].append(svc_results['fscore'][strat][ratio][noise][0])
                svc_precisions_list[strat].append(svc_results['precision'][strat][ratio][noise][0])
                svc_recalls_list[strat].append(svc_results['recall'][strat][ratio][noise][0])
    print(svc_fscores_list, "\n", svc_recalls_list)
    
    


    column_label_map = {
        'no_sample': 'No Sampling',
        'nearest_neighbor': 'Clustering', 
        'undersampling': 'Undersampling',
        #'naive_bayes': 'Naive Bayes',
        'oversampling': 'SMOTE',
        'gan_generated': 'GAN'
    }

    # rf_accuracy = pd.DataFrame.from_dict(rf_accuracies_list, orient='index').transpose().rename(columns=column_label_map)
    print(rf_precisions_list)
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