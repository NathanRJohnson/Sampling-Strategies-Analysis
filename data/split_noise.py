import pandas as pd
import os
from sklearn.model_selection import train_test_split

###############
# splits generated noise data into training and testing sets
# needs to be run from inside /data folder
###############

generated_path = '../data/generated'
# loops through every data file
for fname in os.listdir(generated_path):
    if fname.endswith('.csv'):
        data = pd.read_csv(os.path.join(generated_path, fname))
        
        #splits into training and testing
        train, test = train_test_split(data, test_size=0.30)
        train = train.reset_index().drop(columns='index')
        test = test.reset_index().drop(columns='index')
        
        #saves them in individual files
        fname = fname.split('.csv')[0]
        pd.DataFrame.to_csv(train, f'generated_split/train_{fname}.csv', index=False)
        pd.DataFrame.to_csv(test, f'generated_split/test_{fname}.csv', index=False)