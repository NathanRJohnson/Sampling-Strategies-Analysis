#!/bin/bash  
strat_path=strategies
train_path=data/generated_split
label=label
majority=0
minority=1
out_path=sampled_generated


NN=nearest_neighbor
NB=naive_bayes
RU=undersampling
GN=gan_generated
SM=oversampling


for FILE in $train_path/"train"*;
do  
    IFS='/';
    read -a filename <<< "$FILE";

    mkdir -p data/$out_path/$NN
    python3 $strat_path/$NN.py ${filename[0]}/${filename[1]}/${filename[2]} -l $label -M $majority -m $minority -o data/$out_path/$NN/balanced_${filename[2]}

    #mkdir -p data/$out_path/$NB
    #python3 $strat_path/$NB.py ${filename[0]}/${filename[1]}/${filename[2]} -l $label -M $majority -m $minority -o data/$out_path/$NB/balanced_${filename[2]}

    mkdir -p data/$out_path/$RU
    python3 $strat_path/$RU.py ${filename[0]}/${filename[1]}/${filename[2]} -l $label -M $majority -m $minority -o data/$out_path/$RU/balanced_${filename[2]}

    mkdir -p data/$out_path/$GN
    python3 $strat_path/$GN.py ${filename[0]}/${filename[1]}/${filename[2]} -l $label -M $majority -m $minority -o data/$out_path/$GN/balanced_${filename[2]}

    mkdir -p data/$out_path/$SM
    python3 $strat_path/$SM.py ${filename[0]}/${filename[1]}/${filename[2]} -l $label -M $majority -m $minority -o data/$out_path/$SM/balanced_${filename[2]}


  # mkdir -p $out_path/$NB
  # python3 $strat_path/$NB.py $train_path${i}.csv -l $label -M 3 -m 4 -o $out_path/$NB/balanced_${i}.csv

  # mkdir -p $out_path/$RU
  # python3 $strat_path/$RU.py $train_path${i}.csv -l $label -M 3 -m 4 -o $out_path/$RU/balanced_${i}.csv

  # mkdir -p $out_path/$GN
  # python3 $strat_path/$GN.py $train_path${i}.csv -l $label -M 3 -m 4 -o $out_path/$GN/balanced_${i}.csv

  # mkdir -p $out_path/$SM
  # python3 $strat_path/$SM.py $train_path${i}.csv -l $label -M 3 -m 4 -o $out_path/$SM/balanced_${i}.csv
  done