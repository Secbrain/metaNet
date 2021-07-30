import argparse
import os
from HSUsolver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
import cfg


def main(config):
    cudnn.benchmark = True

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

    train_loader = get_loader(batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train')

    valid_loader = get_loader(batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid')
    test_loader = get_loader(batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test')

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
    parser.add_argument('--model_path', type=str, default=cfg.model_save_path)
    parser.add_argument('--result_path', type=str, default=cfg.result_path)

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
