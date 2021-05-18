#!/bin/bash

# pip install pandas scikit-learn numpy lightgbm
python main.py -task=main -cv=0 -test=1 
# python main.py -task=grid_search
# python main.py -task=data_ablation

# run other models
# python main.py -model=AdaBoost -data_frac=0.1
# python main.py -model=RandomForest -data_frac=0.1
# python main.py -model=MLP -data_frac=0.1