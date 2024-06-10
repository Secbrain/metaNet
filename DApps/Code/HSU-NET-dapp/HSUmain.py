import argparse
import os
from HSUsolver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
import cfg
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

static_command = './dapp_features_with_metafeatures.csv'
num_avg = 1000

def static_100_100():
    vectors = pd.read_csv(static_command)
    
    categories = vectors['category'].unique().tolist()
    category_map = {}
    for i in range(len(categories)):
        category_map[categories[i]] = int(i)
    
    train = []
    valid = []
    test = []

    vector_datas = vectors.values
    vector_datas[:,3:] = StandardScaler().fit_transform(vector_datas[:,3:])
    # vector_datas[:,3:] = MinMaxScaler().fit_transform(vector_datas[:,3:])

    for category in categories:
        vector_datas[vector_datas[:,0] == category, 0] = category_map[category]

    class_type = list(np.unique(vector_datas[:,0]))
    for class_name in class_type:
        data = vector_datas[vector_datas[:,0] == class_name]

        familyType = list(np.unique(data[:,1]))

        num_avg_class = int(num_avg / len(familyType))
        # print(class_name, num_avg_class)

        for family_name in familyType:
            data_1 =  data[data[:,1] == family_name]
            np.random.shuffle(data_1)

            # if data_1.shape[0] < num_avg_class:
            #     times = num_avg_class/data_1.shape[0]
            #     app_data = list(data_1)
            #     app_data = app_data * int(times) + app_data[:int(data_1.shape[0] * (times - int(times)))]
                
            #     np.random.shuffle(app_data)
            #     data_1 = app_data
            # else:
            #     data_1 = data_1[:num_avg_class]
            # data_1 = np.array(data_1)

            train.extend(data_1[:int(data_1.shape[0]*0.8)])
            valid.extend(data_1[int(data_1.shape[0]*0.8):])
            test.extend(data_1[int(data_1.shape[0]*0.8):])

    train = np.array(train)
    valid = np.array(valid)
    test = np.array(test)

    print("train vs test", (np.all(train == valid)))

    # np.random.shuffle(train)
    # np.random.shuffle(valid)
    # np.random.shuffle(test)

    train_data = train[:,3:].astype(float)
    # print(train_data[:10])
    # train_data[:,max_values>1] = train_data[:,max_values>1] / max_values[max_values>1] #* times_max[max_values>1]
    # print(train_data[:10])
    train_label = train[:,0].astype(int)
    train_data = train_data.reshape(-1,4,8,36)

    valid_data = valid[:,3:].astype(float)
    # print(valid_data[:10])
    # valid_data[:,max_values>1] = valid_data[:,max_values>1] / max_values[max_values>1] #* times_max[max_values>1]
    # print(valid_data[:10])
    valid_label = valid[:,0].astype(int)
    valid_data = valid_data.reshape(-1,4,8,36)

    test_data = test[:,3:].astype(float)
    # test_data[:,max_values>1] = test_data[:,max_values>1] / max_values[max_values>1] #* times_max[max_values>1]
    test_label = test[:,0].astype(int)
    test_data = test_data.reshape(-1,4,8,36)


    # kk = vectors.values[:,3:].astype(float)
    # max_values = np.max(kk, axis=0)
    # # max_values[max_values == 0] = 1
    # times_max = np.power(2,np.log10(max_values)) #np.sqrt(max_values) #np.power(2,np.log10(max_values))
    # # min_values = np.min(kk, axis=0)
    # # dis_values = max_values - min_values
    # # dis_values[dis_values == 0] = 1
    
    # # family_out_names = {}
    # class_type = list(vectors['category'].unique())
    # for class_name in class_type:
    #     data = vectors[vectors['category'].isin([class_name])]

    #     familyType = list(data['name'].unique())

    #     num_avg_class = int(num_avg / len(familyType))
    #     # print(class_name, num_avg_class)

    #     for family_name in familyType:
    #         data_1 =  data[data['name'].isin([family_name])].values
    #         np.random.shuffle(data_1)

    #         # if data_1.shape[0] < num_avg_class:
    #         #     times = num_avg_class/data_1.shape[0]
    #         #     app_data = list(data_1)
    #         #     app_data = app_data * int(times) + app_data[:int(data_1.shape[0] * (times - int(times)))]
                
    #         #     np.random.shuffle(app_data)
    #         #     data_1 = app_data
    #         # else:
    #         #     data_1 = data_1[:num_avg_class]
    #         # data_1 = np.array(data_1)

    #         train.extend(data_1[:int(data_1.shape[0]*0.8)])
    #         valid.extend(data_1[int(data_1.shape[0]*0.8):])
    #         test.extend(data_1[int(data_1.shape[0]*0.8):])

    # train = np.array(train)
    # valid = np.array(valid)
    # test = np.array(test)

    # print("train vs test", (np.all(train == valid)))

    # for category in categories:
    #     train[train[:,0] == category, 0] = category_map[category]
    #     valid[valid[:,0] == category, 0] = category_map[category]
    #     test[test[:,0] == category, 0] = category_map[category]

    # # np.random.shuffle(train)
    # # np.random.shuffle(valid)
    # # np.random.shuffle(test)

    # train_data = train[:,3:].astype(float)
    # print(train_data[:10])
    # train_data[:,max_values>1] = train_data[:,max_values>1] / max_values[max_values>1] #* times_max[max_values>1]
    # print(train_data[:10])
    # train_label = train[:,0].astype(int)
    # train_data = train_data.reshape(-1,16,4,18)

    # valid_data = valid[:,3:].astype(float)
    # print(valid_data[:10])
    # valid_data[:,max_values>1] = valid_data[:,max_values>1] / max_values[max_values>1] #* times_max[max_values>1]
    # print(valid_data[:10])
    # valid_label = valid[:,0].astype(int)
    # valid_data = valid_data.reshape(-1,16,4,18)

    # test_data = test[:,3:].astype(float)
    # test_data[:,max_values>1] = test_data[:,max_values>1] / max_values[max_values>1] #* times_max[max_values>1]
    # test_label = test[:,0].astype(int)
    # test_data = test_data.reshape(-1,16,4,18)

    print("train vs test", (np.all(train_data == valid_data)))

    print("train:{}, test:{}".format(len(train_label), len(test_label)))

    return train_data, train_label, valid_data, valid_label, test_data, test_label

def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net', 'HSU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net/HSU_Net')
        print('Your input for model_type was %s' % config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    lr = random.random() * 0.0005 + 0.0000005
    augmentation_prob = random.random() * 0.7
    epoch = random.choice([100, 150, 200, 250])
    decay_ratio = random.random() * 0.8
    decay_epoch = int(epoch * decay_ratio)

    config.augmentation_prob = augmentation_prob

    print(config)

    train_data, train_label, valid_data, valid_label, test_data, test_label = static_100_100()

    train_loader = get_loader(batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              image1=train_data,
                              labels=train_label)

    valid_loader = get_loader(batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              image1=valid_data,
                              labels=valid_label)
    
    test_loader = get_loader(batch_size=config.batch_size,
                             num_workers=config.num_workers,
                              image1=test_data,
                              labels=test_label)

    print(config.output_ch)
    solver = Solver(config, train_loader, valid_loader, test_loader)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # training hyper-parameters
    parser.add_argument('--img1_ch', type=int, default=cfg.img1_ch)
    parser.add_argument('--img2_ch', type=int, default=cfg.img2_ch)
    parser.add_argument('--output_ch', type=int, default=cfg.output_ch)
    parser.add_argument('--num_epochs', type=int, default=cfg.num_epochs)
    parser.add_argument('--num_epochs_decay', type=int, default=cfg.num_epochs_decay)
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=cfg.lr)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='HSU_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/HSU_Net')
    parser.add_argument('--model_path', type=str, default=cfg.model_save_path)
    parser.add_argument('--result_path', type=str, default=cfg.result_path)

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
