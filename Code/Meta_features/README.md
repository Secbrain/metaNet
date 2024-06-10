# Meta-features Extraction

## The running process

The first is to obtain the frequent sets of original features:

```bash
python original_features_to_frequentset.py
```

The second is to extract the matrix of meta-features based on the frequent sets and obtain the final vectors of original features multiplied by it:

```bash
python frequet_features_to_meta_features_matrix.py
```

## Other files introduction

The file "meta_features.npy" refers to the matrix of meta-features.

The files "vectors_features_data.csv“ and "features_value_features.json" include the original feature examples of datasets.

The file "features_name_examples.json" is an example of feature names.

The file "vectors_features_data_with_metafeatures.csv” is the the final vectors of original features based on the "vectors_features_data.csv“.

