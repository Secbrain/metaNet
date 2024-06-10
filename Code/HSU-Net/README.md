# The Running Process

The first step is to generate the files of static features to the folder "dataset2" based on the feature vectors extracted in the meta-feature process:

```bash
python csv_to_npy.py
```

Note that the csv file "vectors_features_data_with_metafeatures.csv" can be viewed in the upper folder "Static_features". Also, the folder "dataset1" includes the files of dynamic features, which are listed in the upper folder "Dynamic_features" and correspond to the folder "dataset2" one by one.

The second step is to divide the training and testing datasets based on the folders "dataset1" and "dataset2":

```bash
python HSUpreprocessor.py
```

The division results of datasets are shown in the folder "dataset1".  Note that the scripts of "HSUpreprocessor_binary.py" and "HSUpreprocessor-family_detail.py" are responsible for the binary classification and multiple classification (training and test samples are divided by family), respectively.

The third step is to train and test the model based on the above datasets.

```bash
python HSUmain.py --mode train
```

The model will be save in the folder "models", and the results will be outputed in the folder "result". Also, the test process can be only performed by executing the following command:

```bash
python HSUmain.py --mode test
```