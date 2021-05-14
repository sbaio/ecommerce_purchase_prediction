#!/bin/bash

# pip install pandas scikit-learn numpy

# python main.py -data_frac=0.1 -model=BoostedTree -drop_nan=1# 0.816+-0.007
# python main.py -data_frac=0.1 -model=AdaBoost -drop_nan=1# 0.815+-0.006
# python main.py -data_frac=1.0 -model=BoostedTree -drop_nan=1# 0.815 0.002
# python main.py -data_frac=0.1 -model=BoostedTree -max_depth=5 -drop_nan=1# 0.817 0.007
# python main.py -data_frac=0.1 -model=BoostedTree -max_depth=10 -drop_nan=1
# python main.py -data_frac=0.1 -model=RandomForest_seippel -drop_nan=1# 0.81455 0.007
# python main.py -data_frac=0.1 -model=LogisticRegression -verbose=2 -drop_nan=1# 0.746 0.006
# python main.py -data_frac=0.1 -model=RandomForest -drop_nan=1# ~ 0.66
## Added categorical variables as preprocessing
# python main.py -data_frac=0.1 -model=BoostedTree -verbose=2 -drop_nan=1# 0.832 0.009
## Do not drop nan values
# python main.py -data_frac=0.1 -model=BoostedTree -verbose=2 -drop_nan=0 # 0.848 0.005
## feature selection
python main.py -data_frac=0.1 -n_estimators=500 -model=BoostedTree -verbose=2 -drop_nan=0 -max_cv_runs=3 -outdir=out/
# python main.py -data_frac=1.0 -n_estimators=500 -model=BoostedTree -verbose=2 -drop_nan=0 -max_cv_runs=3 -outdir=out/