

# covertype data preprocessing
cd data
echo 'Cleaning Covertype data'
python3 clean.py

echo 'Generating Covertype folds'
python3 fold.py

cd ..
# run sampling strategies
echo Creating balanced datasets
chmod +x run_strategies.sh
./run_strategies.sh

# experiment
echo Running experiment
python3 experiment.py

# creating figures
echo Generating figures
python3 boxplot.py

# creating table
echo Generating tables
python3 table.py

echo 
echo Performing noise Analysis
cd data 

echo Generating noise data
python3 generator.py

echo Splitting noise data
python3 split_noise.py

cd ..
echo Sampling noise data
chmod +x run_strategies_generated.sh
./run_strategies_generated.sh

echo running noise training and testing
python3 experiment_generated.py

echo generating boxplots
python3 box_plot_generated.py

echo all done!