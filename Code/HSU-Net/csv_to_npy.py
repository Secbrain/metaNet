import os
import random

import cfg
import numpy as np
import glob
import numpy as np
import pandas as pd
import io

root_path = "./dataset2/"

csv_features_path = 'vectors_features_data_with_metafeatures.csv'

def csv_to_npy():
    csvfile = io.open(csv_features_path,'r',encoding="utf-8")
    df = pd.read_csv(csvfile,header=None)
    csvfile.close()

    for line in range(len(df)):
        test_text = df.iloc[line,3:].values

        class_name = df.iloc[line,1]
        familie_name = df.iloc[line,2]
        
        sample_dir = os.path.join(root_path, class_name, familie_name)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        np.save(os.path.join(sample_dir, "{}.npy".format(df.iloc[line,0])), test_text)

csv_to_npy()