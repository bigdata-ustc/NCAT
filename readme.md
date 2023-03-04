### Prepare and preprocess dataset
Put the dataset (e.g., ASSIST: 2012-2013-data-with-predictions-4-final.csv) into the path data/[data_name]/
```
cd data/[data_name]
python data_preprocess.py
```

### Dividing the dataset (Train Test Valid)
```
python divide_data.py
```

### CDM Training
```
cd pre_train
python main.py
```

### Model Training
```
sh ./model_train.sh
```