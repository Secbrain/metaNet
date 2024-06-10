# File and Folder introduction

## pcap_to_features_csv.py

This file aims to extract the feature vectors based on the pcap files.

Furthermore, the meta-features of DApps can be obtained by leveraging the same method with the Malware dataset. The final feature vectors can be viewed in the upper folder "Feature".

## HSU-NET-dapp

This folder includes the model code of HSU-Net for the DApp identification. Different from the Malware dataset, this model can be run by executing the following command directly:

```bash
python HSUmain.py
```

Note that the csv file "dapp_features_with_metafeatures.csv" can be viewed in the upper folder "Feature".