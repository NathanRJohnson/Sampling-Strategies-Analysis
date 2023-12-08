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


##################################
# GAN for covertype datas3t
# same as GAN for generated dataset but with slightly different values to fit the data
##################################

#function to parse the command line arguments
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

# originally based on tutorial from https://realpython.com/generative-adversarial-networks/ 
# but layers, activation functions, etc. have been changed to match 
# discriminator is used to predict if a data point is real or generated
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

# generator is used to create new data points
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
    

#prepares data for the model
def prepData(data, labels, batch_size, scaler):
    labels = np.array(labels)
    data = scaler.fit_transform(data)
    dataset = [(data[i], labels[i]) for i in range(len(data))]
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    path_to_input, inputs=parse_args()

    #random generator seed - used so the results can be recreated even with random generation
    torch.manual_seed(10)

    #read data from csv, extract minority class data
    data = pd.read_csv(path_to_input)
    minority_class_data = data[data[inputs.label] == inputs.minority].reset_index(drop=True)

    labels = data[inputs.label]
    minority_class_labels = minority_class_data[inputs.label]

    data = data.drop(columns=[inputs.label])
    data = np.array(data)

    minority_class_data = minority_class_data.drop(columns=[inputs.label])

    all_train_x, all_test_x, all_train_labels, all_test_labels = train_test_split(data, labels, test_size = 0.25, random_state=10)
    minority_train_x, minority_test_x, minority_train_labels, minority_test_labels = train_test_split(minority_class_data, minority_class_labels, test_size = 0.25, random_state=10)

    #determining batch size based on the size of the train set
    batch_size = 50
    while(len(minority_train_x)%batch_size != 0):
        batch_size+=1

    #preprocessing data 
    minority_train_x = np.array(minority_train_x)
    minority_test_x = np.array(minority_test_x)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_x = prepData(minority_train_x, minority_train_labels, batch_size, scaler)
    test_x = prepData(minority_test_x, minority_test_labels, batch_size, scaler)
    
    #initialize discriminator and generator
    disc = Discriminator()
    gen = Generator()

    #set parameters
    num_of_batches = len(train_x)
    lr = 0.0001
    epochs = 200
    loss_fn = nn.BCELoss()
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr)
    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr)

    #training discriminator and generator
    for e in range(epochs):
        for n, (true_data,i)in enumerate(train_x):
            true_data = true_data.float()
            
            # generate data using generator to use to train discriminator
            true_data_labels = torch.zeros((batch_size, 1))
            sample_spaces = torch.randn((batch_size, np.shape(data)[1]))
            gen_data = gen(sample_spaces)
            gen_data_labels = torch.ones((batch_size, 1))
            all_data = torch.cat((true_data, gen_data))
            all_data_labels = torch.cat(
                (true_data_labels, gen_data_labels)
            )

            # train the discriminator
            disc.zero_grad()
            output_disc = disc(all_data)
            loss_disc = loss_fn(
                output_disc, all_data_labels)
            loss_disc.backward()
            opt_disc.step()

            # create empty space for generator
            sample_spaces = torch.randn((batch_size, np.shape(data)[1]))

            # train the generator
            gen.zero_grad()
            gen_data = gen(sample_spaces)
            output_disc_gen = disc(gen_data)
            loss_gen = loss_fn(
                output_disc_gen, true_data_labels
            )
            loss_gen.backward()
            opt_gen.step()

            # print loss for discriminator and generator
            # used to evaluate - trying to balance disc loss and gen loss
            if e % 10 == 0 and n == num_of_batches - 1:
                print(f"Epoch: {e} Loss D.: {loss_disc}")
                print(f"Epoch: {e} Loss G.: {loss_gen}")



    # balancing the dataset - generate data until the dataset is even
    # not exactly same amount in minority and majority because it is generated in batches. 
    sample_spaces = torch.randn((batch_size, np.shape(data)[1]))
    gen_data = gen(sample_spaces)
    while len(labels[labels == inputs.minority])< len(labels[labels == inputs.majority]):
        sample_spaces = torch.randn((batch_size, np.shape(data)[1]))
        gen_data = gen(sample_spaces)
        gen_data = scaler.inverse_transform(gen_data.detach().numpy())
        labels = pd.concat([labels,pd.Series(inputs.minority, index=range(len(gen_data)))])
        data = np.concatenate((data,gen_data))
    data_df = pd.DataFrame(data, columns =minority_class_data.columns)
    data_df[inputs.label] = labels.values

    #save data to csv
    data_df.to_csv(inputs.output, index=False)
    


if __name__ == '__main__':
    main()