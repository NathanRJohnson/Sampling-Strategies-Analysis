import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# turns off too many plots warning
plt.rcParams.update({'figure.max_open_warning': 0})

# makes plot for each metric
def make_plots(precision, recall, fscores, delta_fscores, clf):
    if clf == 'Random Forest':
        clf_path = clf.lower().replace(' ', '_')
    else:
        clf_path = "svc"
        
    #precision plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=precision)
    plt.xlabel('Strategy')
    plt.ylabel('Precision')
    plt.title(f'Precision by Strategy - {clf}')

    plt.savefig(f'results/{clf_path}_generated/{clf}_precision_boxplot_generated.png')

    plt.clf()

    #recall plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=recall)
    plt.xlabel('Strategy')
    plt.ylabel('Recall')
    plt.title(f'Recall by Strategy - {clf}')

    plt.savefig(f'results/{clf_path}_generated/{clf}_recall_boxplot_generated.png')

    plt.clf()
    
    #f score plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=fscores)
    plt.xlabel('Strategy')
    plt.ylabel('F Score')
    plt.title(f'F Score by Strategy - {clf}')

    plt.savefig(f'results/{clf_path}_generated/{clf}_fscore_boxplot_generated.png')
    
    #delta f score plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=delta_fscores)
    plt.xlabel('Strategy')
    plt.ylabel('Delta F Score')
    plt.title(f'Delta F Score by Strategy - {clf}')

    plt.savefig(f'results/{clf_path}_generated/{clf}_delta_fscore_boxplot_generated.png')
    plt.close()
 
#creates plots for individual strategies to show the results of the different noise levels   
def plot_by_noise(precision, recall, fscore, fscore_nosample, clf, strat):
    if clf == 'Random Forest':
        clf_path = clf.lower().replace(' ', '_')
    else:
        clf_path = "svc"
    
    #initializing lists
    precision_list = {}
    recall_list = {}
    fscore_list = {}
    delta_fscore_list = {}

    #looping through layers of json
    for ratio in precision:
        for noise in precision[ratio]:
            if noise not in precision_list.keys():
                precision_list[noise] = []
                recall_list[noise] = []
                fscore_list[noise] = []
                delta_fscore_list[noise] = []
            precision_list[noise].append(precision[ratio][noise][0])
            recall_list[noise].append(recall[ratio][noise][0])
            fscore_list[noise].append(fscore[ratio][noise][0])

            delta_fscore_list[noise].append(fscore[ratio][noise][0] - fscore_nosample[ratio][noise][0])
    
    precision_df = pd.DataFrame.from_dict(precision_list, orient='index').transpose()
    recall_df = pd.DataFrame.from_dict(recall_list, orient='index').transpose()
    fscore_df = pd.DataFrame.from_dict(fscore_list, orient='index').transpose()
    delta_fscore_df = pd.DataFrame.from_dict(delta_fscore_list, orient='index').transpose()

    #precision plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=precision_df.loc[:,['0','5','10','20']])
    plt.xlabel('Noise Level')
    plt.ylabel('Precision')
    plt.title(f'Precision by Noise Level - {strat} {clf}')

    plt.savefig(f'results/{clf_path}_generated/{clf}_{strat}_precision_boxplot_generated.png')
    plt.clf()
    
    #recall plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=recall_df.loc[:,['0','5','10','20']])
    plt.xlabel('Noise Level')
    plt.ylabel('Recall')
    plt.title(f'Recall by Noise Level - {strat} {clf}')

    plt.savefig(f'results/{clf_path}_generated/{clf}_{strat}_recall_boxplot_generated.png')
    plt.clf()
    
    #f score plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=fscore_df.loc[:,['0','5','10','20']])
    plt.xlabel('Noise Level')
    plt.ylabel('F Score')
    plt.title(f'F Score by Noise Level - {strat} {clf}')

    plt.savefig(f'results/{clf_path}_generated/{clf}_{strat}_fscore_boxplot_generated.png')
    plt.clf()
    
 # delta f score plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=delta_fscore_df.loc[:,['0','5','10','20']])
    plt.xlabel('Noise Level')
    plt.ylabel('Delta F Score')
    plt.title(f'Delta F Score by Noise Level - {strat} {clf}')

    plt.savefig(f'results/{clf_path}_generated/{clf}_{strat}_delta_fscore_boxplot_generated.png')

    plt.clf()
    plt.close()


def main():
    #get metrics for random forest
    with open('results/random_forest_generated/metrics.json', 'r') as file:
        rf_results = json.load(file)
    
    #initialize list for each metric
    rf_fscores_list = {}
    rf_delta_fscores_list = {}
    rf_precisions_list = {}
    rf_recalls_list = {}
    #looping through the layers of the json - strategy, ratio of majority to minority, noise level
    for strat in rf_results['fscore']:
        rf_fscores_list[strat] = []
        rf_delta_fscores_list[strat] = []
        rf_precisions_list[strat] = []
        rf_recalls_list[strat] = []
        plot_by_noise(rf_results['precision'][strat], rf_results['recall'][strat], rf_results['fscore'][strat], rf_results['fscore']['no_sample'], "Random Forest", strat)
        for ratio in rf_results['fscore'][strat]:
            for noise in rf_results['fscore'][strat][ratio]:
                rf_fscores_list[strat].append(rf_results['fscore'][strat][ratio][noise][0])
                rf_precisions_list[strat].append(rf_results['precision'][strat][ratio][noise][0])
                rf_recalls_list[strat].append(rf_results['recall'][strat][ratio][noise][0])
                #delta f score is the difference between fscore of balanced data and fscore of unbalanced data
                if (strat != "no_sample"):
                    rf_delta_fscores_list[strat].append(rf_results['fscore'][strat][ratio][noise][0]-rf_results['fscore']['no_sample'][ratio][noise][0]) 


    #get metics for svc 
    with open('results/svc_generated/metrics.json', 'r') as file:
        svc_results = json.load(file)
    
    #initialize list for each metric 
    svc_fscores_list = {}
    svc_delta_fscores_list = {}
    svc_precisions_list = {}
    svc_recalls_list = {}
    
    #loop through layers of json - strat, ratio, noise level
    for strat in svc_results['fscore']:
        svc_fscores_list[strat] = []
        svc_delta_fscores_list[strat] = []
        svc_precisions_list[strat] = []
        svc_recalls_list[strat] = []
        plot_by_noise(svc_results['precision'][strat], svc_results['recall'][strat], svc_results['fscore'][strat], svc_results['fscore']['no_sample'], "Support Vector Machine", strat)
        for ratio in svc_results['fscore'][strat]:
            for noise in svc_results['fscore'][strat][ratio]:
                svc_fscores_list[strat].append(svc_results['fscore'][strat][ratio][noise][0])
                svc_precisions_list[strat].append(svc_results['precision'][strat][ratio][noise][0])
                svc_recalls_list[strat].append(svc_results['recall'][strat][ratio][noise][0])
                if (strat != "no_sample"):
                    svc_delta_fscores_list[strat].append(svc_results['fscore'][strat][ratio][noise][0]-svc_results['fscore']['no_sample'][ratio][noise][0]) 

    
    


    column_label_map = {
        'no_sample': 'No Sampling',
        'nearest_neighbor': 'Clustering', 
        'undersampling': 'Undersampling',
        'naive_bayes': 'Naive Bayes',
        'oversampling': 'SMOTE',
        'gan_generated': 'GAN'
    }

    rf_precision = pd.DataFrame.from_dict(rf_precisions_list, orient='index').transpose().rename(columns=column_label_map)
    rf_recall = pd.DataFrame.from_dict(rf_recalls_list, orient='index').transpose().rename(columns=column_label_map)
    rf_fscore = pd.DataFrame.from_dict(rf_fscores_list, orient='index').transpose().rename(columns=column_label_map)
    rf_delta_fscore = pd.DataFrame.from_dict(rf_delta_fscores_list, orient='index').transpose().rename(columns=column_label_map)


    make_plots(rf_precision, rf_recall, rf_fscore, rf_delta_fscore, "Random Forest")

    svc_fscore = pd.DataFrame.from_dict(svc_fscores_list, orient='index').transpose().rename(columns=column_label_map)
    svc_precision = pd.DataFrame.from_dict(svc_precisions_list, orient='index').transpose().rename(columns=column_label_map)
    svc_recall = pd.DataFrame.from_dict(svc_recalls_list, orient='index').transpose().rename(columns=column_label_map)
    svc_delta_fscore = pd.DataFrame.from_dict(svc_delta_fscores_list, orient='index').transpose().rename(columns=column_label_map)


    make_plots(svc_precision, svc_recall, svc_fscore, svc_delta_fscore, "Support Vector Machine")

if __name__ == '__main__':
    main()