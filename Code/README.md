# File and Folder introduction

## Dynamic_features
This folder shows the dynamic features we extracted.

## Static_features
This folder shows the static features we extracted.

## Script
This folder shows some script files we used.

## Meta_features
This folder includes the code for extracting meta-features.

## HSU-Net
This folder contains the code of HSU-Net.

# The Running Process

This first step is to obtain the static and dynamic features according to the methods described in the paper. Their examples are listed in the folders "Dynamic_features" and "Static_features", respectively.

The second step is to extract the meta-features of static features and obtain the final static feature vectors, which is detailed in the folder "Meta_features".

The third step is to perform the model training and testing based on the geneted datasets, which is detailed in the folder "HSU-Net". Note that the HSU-Net with only one input type is different and illustrated in the upper folder "DApps".