import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch 
from torch import nn
import math
import matplotlib.pyplot as plt
from scipy import stats



#https://realpython.com/generative-adversarial-networks/ 
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(53, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, data):
        return self.model(data)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(53, 106),
            nn.ReLU(),
            nn.Linear(106, 212),
            nn.ReLU(),
            nn.Linear(212, 53),
        )
        
    def forward(self, data):
        return self.model(data)
    

def prepData(data, labels, batch_size):
    data = np.array(data)
    data = preprocessing.normalize(data)
    labels = np.array(labels)
    dataset = [(data[i], labels[i]) for i in range(len(data))]
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


#random generator seed - used so the results can be recreated even with random generation
torch.manual_seed(10)

WA_columns = ['wilderness_A_' + str(x + 1) for x in range(4)]
soil_columns = ['soil_T_' + str(x + 1) for x in range(40)]

columns = ["elevation", "aspect", "slope", "hydro_horizontal_dist", "hydro_vertical_dist", "road_horizontal_dist" "shade_9am", "shade_noon", "shade_3pm", "firepoints_horizontal_dist"] + WA_columns + soil_columns + ["cover_type"]

whole_dataset = pd.read_csv('./data/raw/covtype.data', header=None, names=columns)

subset_data = pd.concat([whole_dataset[whole_dataset['cover_type'] == 3], whole_dataset[whole_dataset['cover_type'] == 4]]).reset_index().drop(columns=['index'])

print("# of samples with covertype Ponderosa Pine: ", len(subset_data[subset_data['cover_type'] == 3]))
print("# of samples with covertype Cottonwood/Willow: ", len(subset_data[subset_data['cover_type'] == 4]))

subset_labels = subset_data["cover_type"]
subset_data = subset_data.drop(columns=['cover_type'])
train_x, test_x, train_labels, test_labels = train_test_split(subset_data, subset_labels, test_size = 0.2, random_state=10)

batch_size = 100
train_x = prepData(train_x, train_labels, batch_size)
test_x = prepData(test_x, test_labels, batch_size)

##tutorial
'''
train_data_length = 1024
train_data = torch.zeros((train_data_length, 2))
train_data[:,0] = 2*math.pi*torch.rand(train_data_length)
train_data[:, 1] = torch.sin(train_data[:, 0])
train_labels = torch.zeros(train_data_length)
train_set = [
    (train_data[i], train_labels[i]) for i in range(train_data_length)
]

#batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

#print(train_set)
'''
print("length of train", len(train_x))
disc = Discriminator()
gen = Generator()

lr = 0.0005
epochs = 100
loss_fn = nn.BCELoss()
opt_disc = torch.optim.Adam(disc.parameters(), lr=lr)
opt_gen = torch.optim.Adam(gen.parameters(), lr=lr)

for e in range(epochs):
    for n, (true_data,i)in enumerate(train_x):
        true_data = true_data.float()
        # Data for training the discriminator
        true_data_labels = torch.ones((batch_size, 1))
        sample_spaces = torch.randn((batch_size, 53))
        gen_data = gen(sample_spaces)
        gen_data_labels = torch.zeros((batch_size, 1))
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
        sample_spaces = torch.randn((batch_size, 53))

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
        if e % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {e} Loss D.: {loss_disc}")
            print(f"Epoch: {e} Loss G.: {loss_gen}")



#Evaluating performance of generator
#Creates a batch of samples using generator then compares it to batches of real data.
#Compares each column/feature using the ttest. Plots the p value on a graph with 0.05 line drawn
sample_spaces = torch.randn((batch_size, 53))
gen_data = gen(sample_spaces)
gen_data = gen_data.detach()
sample, i = next(iter(train_x))

difference = pd.DataFrame(columns=subset_data.columns)
for n, (true_data,i)in enumerate(train_x):
    newrow = {}
    for col in range(len(subset_data.columns)):
        t, p = stats.ttest_ind(sample[col], true_data[col])
        newrow[subset_data.columns[col]] = [p]
    difference = pd.concat([difference, pd.DataFrame.from_dict(newrow)])
difference = difference.reset_index().drop(columns=['index'])
plt.plot(difference.sum()/len(train_x))
plt.axhline(y = 0.05, color = 'r', linestyle = '-')
plt.xticks(rotation=90)
plt.show()


'''
fig, ax = plt.subplots(2, 3)
ax[0,0].scatter(gen_data[:,0], gen_data[:,1])
ax[0,0].scatter(sample[:,0], sample[:,1])
ax[0,0].legend(['Generated', 'real'])

ax[1,0].scatter(gen_data[:,2], gen_data[:,3])
ax[1,0].scatter(sample[:,2], sample[:,3])
ax[1,0].legend(['Generated', 'real'])

ax[0,1].scatter(gen_data[:,4], gen_data[:,5])
ax[0,1].scatter(sample[:,4], sample[:,5])
ax[0,1].legend(['Generated', 'real'])

ax[1,1].scatter(gen_data[:,6], gen_data[:,7])
ax[1,1].scatter(sample[:,6], sample[:,7])
ax[1,1].legend(['Generated', 'real'])

ax[0,2].scatter(gen_data[:,2], gen_data[:,3])
ax[0,2].scatter(sample[:,2], sample[:,3])
ax[0,2].legend(['Generated', 'real'])

plt.show()

'''