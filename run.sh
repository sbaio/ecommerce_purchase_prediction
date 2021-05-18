#!/bin/bash

# pip install pandas scikit-learn numpy lightgbm
python script.py -task=main -cv=0 -test=1 
# python script.py -task=grid_search
# python script.py -task=data_ablation

# run other models
# python script.py -model=AdaBoost -data_frac=0.1
# python script.py -model=RandomForest -data_frac=0.1
# python script.py -model=MLP -data_frac=0.1