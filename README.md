# Asos purchase probability prediction

## Requirements
```
pip install pandas scikit-learn numpy lightgbm
```

## Setting up data
Setup --data_dir to point to the folder containing the data.
It is possible to setup --cache_dir to a directory where to save cached data.


## Training a GBDT model
The notebook data_analysis.ipynb shows the steps of training a boosted trees model using the lightgbm library and predicting on the test set.

Also we can run the training and inference on the test data by using 
```
python main.py -task=main -cv=0 -test=1 
```

## Running a grid search
```
python main.py -task=grid_search
```

## Running a data ablation
We compare the use of different sources of features (customer, product, views ...)

```
python main.py -task=data_ablation
```
