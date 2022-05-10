# kaggle-h-and-m-personalized-fashion-recommendations

https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations

## preparation
### prepare competition data
```console
kaggle competitions download -c h-and-m-personalized-fashion-recommendations
unzip h-and-m-personalized-fashion-recommendations.zip -d input/h-and-m-personalized-fashion-recommendations
```

### requirements
see `setup.sh`

### precalculation
```console
./prepare.sh
```
- renumber input data
- create LightFM embeddings
- calculate user one hot features

## create single model submission
- execute `local6.ipynb` and get `submission.csv`

It takes ~15h with 64 core & 192GB RAM machine

private score is 0.03391

## create 11th place submission
final submission is blending of two local best notebooks and two public best notebooks

weights are determined from validation results

- execute `local6.ipynb` and `mv submission.csv submissions/local6.csv`
- execute `local8.ipynb` and `mv submission.csv submissions/local8.csv`
- execute `public8.ipynb` and `mv submission.csv submissions/public8.csv`
- execute `public12.ipynb` and `mv submission.csv submissions/public12.csv`
- execute `python final_blender.py` and get `submission.csv`

private score is 0.03414
