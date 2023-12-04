strat_path=strategies
train_path=data/folded/train_fold_
label=Cover_Type
majority=3
minority=4
out_path=data/sampled


NN=nearest_neighbor
NB=naive_bayes
RU=undersampling


for i in {0..4}
do
  mkdir -p $out_path/$NN
  python3 $strat_path/$NN.py $train_path${i}.csv -l $label -M 3 -m 4 -o $out_path/$NN/balanced_${i}.csv

  # mkdir -p $out_path/$NB
  # python3 $strat_path/$NB.py $train_path${i}.csv -l $label -M 3 -m 4 -o $out_path/$NB/balanced_${i}.csv

  # mkdir -p $out_path/$RU
  # python3 $strat_path/$RU.py $train_path${i}.csv -l $label -M 3 -m 4 -o $out_path/$RU/balanced_${i}.csv
  done