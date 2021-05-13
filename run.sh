#!/bin/bash

# pip install pandas scikit-learn numpy

# python main.py -data_frac=0.1 -model=BoostedTree # 0.816+-0.007
# python main.py -data_frac=0.1 -model=AdaBoost # 0.815+-0.006
# python main.py -data_frac=1.0 -model=BoostedTree # 0.815 0.002
# python main.py -data_frac=0.1 -model=BoostedTree -max_depth=5 # 0.817 0.007
# python main.py -data_frac=0.1 -model=BoostedTree -max_depth=10
# python main.py -data_frac=0.1 -model=RandomForest_seippel # 0.81455 0.007
# python main.py -data_frac=0.1 -model=LogisticRegression -verbose=2 # 0.746 0.006
# python main.py -data_frac=0.1 -model=RandomForest # ~ 0.66
## Added categorical variables as preprocessing
python main.py -data_frac=0.1 -model=BoostedTree -verbose=2