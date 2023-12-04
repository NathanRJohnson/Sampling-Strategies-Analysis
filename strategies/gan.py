import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch 
from torch import nn
import math
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import sys
import argparse

def parse_args():
    path_to_input = sys.argv[1]
    flag_args = sys.argv[2:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', help='Column name of the label')
    parser.add_argument('-m', '--minority', type=int, help="label value of the minority class")
    parser.add_argument('-M', '--majority', type=int, help="label value of the majority class")
    parser.add_argument('-o', '--output', help='File name for the ouput class')
    args = parser.parse_args(flag_args)
    return path_to_input, args

#https://realpython.com/generative-adversarial-networks/ 
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(27, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, data):
        return self.model(data)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(27, 106),
            nn.ReLU(),
            nn.Linear(106, 212),
            nn.ReLU(),
            nn.Linear(212, 27),
        )
        
    def forward(self, data):
        return self.model(data)
    

def prepData(data, labels, batch_size, scaler):
    labels = np.array(labels)
    data = scaler.fit_transform(data)
    dataset = [(data[i], labels[i]) for i in range(len(data))]
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    path_to_input, inputs=parse_args()

    #random generator seed - used so the results can be recreated even with random generation
    torch.manual_seed(10)

    #Creating the dataframe to hold the data from covtype.data
    # WA_columns = ['wilderness_A_' + str(x + 1) for x in range(4)]
    # soil_columns = ['soil_T_' + str(x + 1) for x in range(40)]

    # columns = ["elevation", "aspect", "slope", "hydro_horizontal_dist", "hydro_vertical_dist", "road_horizontal_dist" "shade_9am", "shade_noon", "shade_3pm", "firepoints_horizontal_dist"] + WA_columns + soil_columns + ["cover_type"]

    # whole_dataset = pd.read_csv(path_to_input, header=None, names=columns)

    subset_data = pd.read_csv(path_to_input)
    minority_class_data = subset_data[subset_data[inputs.label] == inputs.minority].reset_index(drop=True)

    print("# of samples with ",inputs.label, " ", inputs.minority, ": ", len(subset_data[subset_data[inputs.label] == inputs.minority]))
    print("# of samples with ", inputs.label, " ", inputs.majority, ": ", len(subset_data[subset_data[inputs.label] == inputs.majority]))

    subset_labels = subset_data[inputs.label]
    minority_class_labels = minority_class_data[inputs.label]

    subset_data = subset_data.drop(columns=[inputs.label])
    subset_data = np.array(subset_data)
    #subset_data = preprocessing.normalize(subset_data)
    minority_class_data = minority_class_data.drop(columns=[inputs.label])

    all_train_x, all_test_x, all_train_labels, all_test_labels = train_test_split(subset_data, subset_labels, test_size = 0.25, random_state=10)
    minority_train_x, minority_test_x, minority_train_labels, minority_test_labels = train_test_split(minority_class_data, minority_class_labels, test_size = 0.25, random_state=10)

    #random forest classifier
    randForest = RandomForestClassifier()
    randForest.fit(all_train_x, all_train_labels)
    imbalanced_random_forest_score = randForest.score(all_test_x, all_test_labels)
    print("size of train: ", len(minority_train_x))
    batch_size = 50
    while(len(minority_train_x)%batch_size != 0):
        batch_size+=1
    print(batch_size)
    minority_train_x = np.array(minority_train_x)
    minority_test_x = np.array(minority_test_x)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_x = prepData(minority_train_x, minority_train_labels, batch_size, scaler)
    test_x = prepData(minority_test_x, minority_test_labels, batch_size, scaler)
    disc = Discriminator()
    gen = Generator()

    num_of_batches = len(train_x)
    print("number of batches: ", len(train_x))
    lr = 0.0001
    epochs = 200
    loss_fn = nn.BCELoss()
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr)
    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr)

    for e in range(epochs):
        for n, (true_data,i)in enumerate(train_x):
            true_data = true_data.float()
            # Data for training the discriminator
            true_data_labels = torch.zeros((batch_size, 1))
            sample_spaces = torch.randn((batch_size, np.shape(subset_data)[1]))
            gen_data = gen(sample_spaces)
            gen_data_labels = torch.ones((batch_size, 1))
            all_data = torch.cat((true_data, gen_data))
            all_data_labels = torch.cat(
                (true_data_labels, gen_data_labels)
            )

            # Training the discriminator
            disc.zero_grad()
            output_disc = disc(all_data)
            loss_disc = loss_fn(
                output_disc, all_data_labels)
            loss_disc.backward()
            opt_disc.step()

            # Data for training the generator
            sample_spaces = torch.randn((batch_size, np.shape(subset_data)[1]))

            # Training the generator
            gen.zero_grad()
            gen_data = gen(sample_spaces)
            output_disc_gen = disc(gen_data)
            loss_gen = loss_fn(
                output_disc_gen, true_data_labels
            )
            loss_gen.backward()
            opt_gen.step()

            # Show loss
            if e % 10 == 0 and n == num_of_batches - 1:
                print(f"Epoch: {e} Loss D.: {loss_disc}")
                print(f"Epoch: {e} Loss G.: {loss_gen}")



    #Evaluating performance of generator
    #Creates a batch of samples using generator then compares it to batches of real data.
    #Compares each column/feature using the ttest. Plots the p value on a graph with 0.05 line drawn
    sample_spaces = torch.randn((batch_size, np.shape(subset_data)[1]))
    gen_data = gen(sample_spaces)
    while len(subset_labels[subset_labels == inputs.minority])< len(subset_labels[subset_labels == inputs.majority]):
        sample_spaces = torch.randn((batch_size, np.shape(subset_data)[1]))
        gen_data = gen(sample_spaces)
        gen_data = scaler.inverse_transform(gen_data.detach().numpy())
        subset_labels = pd.concat([subset_labels,pd.Series(inputs.minority, index=range(len(gen_data)))])
        subset_data = np.concatenate((subset_data,gen_data))
    subset_data_df = pd.DataFrame(subset_data, columns =minority_class_data.columns)
    subset_data_df[inputs.label] = subset_labels.values
    subset_data = subset_data[:len(subset_labels[subset_labels == inputs.majority])*2] #removing the excess minority samples created.
    print(inputs.minority, ": ", len(subset_data_df[subset_data_df[inputs.label] == inputs.minority]), inputs.majority,": ", len(subset_data_df[subset_data_df[inputs.label] == inputs.majority]))
    
    subset_data_df.to_csv(inputs.output, index=False)
    
    '''
    #testing with randomforest
    new_train_x, new_test_x,  = train_test_split(subset_data_df, test_size = 0.25, random_state=10)
    randForest = RandomForestClassifier()
    randForest.fit(new_train_x.drop(columns=inputs.label), new_train_x[inputs.label])
    print("Random forest classifier score before balancing: ", imbalanced_random_forest_score)
    print("Random forest classifier score after balancing: ",randForest.score(new_test_x.drop(columns=inputs.label), new_test_x[inputs.label]))
    
    
    
    
    print(len(gen_data))
    sample, i = next(iter(train_x)) 
    sample = scaler.inverse_transform(sample)
    
    fig, ax = plt.subplots(2, 2)
    ax[0,0].scatter(gen_data[:,0], gen_data[:,1])
    ax[0,0].scatter(sample[:,0], sample[:,1])
    ax[0,0].set_xlabel(columns[0])
    ax[0,0].set_ylabel(columns[1])
    ax[0,0].legend(['Generated', 'Real'])

    ax[1,0].scatter(gen_data[:,2], gen_data[:,3])
    ax[1,0].scatter(sample[:,2], sample[:,3])
    ax[1,0].set_xlabel(columns[2])
    ax[1,0].set_ylabel(columns[3])
    ax[1,0].legend(['Generated', 'Real'])

    ax[0,1].scatter(gen_data[:,4], gen_data[:,5])
    ax[0,1].scatter(sample[:,4], sample[:,5])
    ax[0,1].set_xlabel(columns[4])
    ax[0,1].set_ylabel(columns[5])
    ax[0,1].legend(['Generated', 'Real'])

    ax[1,1].scatter(gen_data[:,6], gen_data[:,7])
    ax[1,1].scatter(sample[:,6], sample[:,7])
    ax[1,1].set_xlabel(columns[6])
    ax[1,1].set_ylabel(columns[7])
    ax[1,1].legend(['Generated', 'Real'])

    fig.suptitle("Real Vs. Generated Data")
    fig.tight_layout()
    plt.show()
    '''

if __name__ == '__main__':
    main()