# kaggle-h-and-m-personalized-fashion-recommendations

https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations

## preparation
download and extract competition data
```
kaggle competitions download -c h-and-m-personalized-fashion-recommendations
unzip h-and-m-personalized-fashion-recommendations.zip -d input/h-and-m-personalized-fashion-recommendations
```

create renumbered data
```
python transform_data.py
```

## create submission
1. create csv file with `customer_id_idx` and `prediction_idx`
1. execute `python transform_submission.py <input csv> <output csv>`
