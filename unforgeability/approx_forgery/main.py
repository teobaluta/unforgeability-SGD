import argparse
import math
import multiprocessing as mp
import os
import pickle
import random
import time
import timeit
from collections import defaultdict

import approx_forgery.helpers as helpers
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import train.data as train_data
import train.utils as utils
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

random.seed(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float64)
torch.use_deterministic_algorithms(True)
rng = np.random.default_rng(42)

parser = argparse.ArgumentParser()

parser.add_argument('--runs', type=int, help='number of independent runs.')
parser.add_argument('--ckpt_dir', type=str, help='directory where checkpoints are stored.')
parser.add_argument('--out_dir', type=str, help='directory where results are stored.')
parser.add_argument('--seed', type=int, help='Seed to initialize randomness')
parser.add_argument('--arch', type=utils.Architecture, default=utils.Architecture.LeNet5,
                    help='Type of model architecture.')
parser.add_argument('--dataset', type=utils.Dataset, default=utils.Dataset.MNIST,
                    help='The name of the dataset.')
parser.add_argument('--batch_size', type=int, help='batch_size for forging')
parser.add_argument('--num_epochs', type=int, help='number of epochs for divergence')
parser.add_argument('--device', type=str, help='which gpu to run on')
parser.add_argument('--mode', type=str, help='which experiment to run')
parser.add_argument('--norm', type=str, help='which norm to use for forging')
parser.add_argument('--epoch', type=int, help='which epoch to use for shuffle or shuffle divergence')
parser.add_argument('--ts', type=int, help='which ts to use for shuffle or shuffle divergence')
parser.add_argument('--training_steps', type=int, help='till which training step to plot the divergence error for')
parser.add_argument('--num_shuffles', type=int, help='number of orderings to take for commutativity test')
parser.add_argument('--num_forging_candidates', type=int,
                    help='number of candidate batches to take for forging, M argument in the prior work paper')
parser.add_argument('--forgery_dir', type=str, help='directory where forged checkpoints are stored')
parser.add_argument('--ckpt_name', type=str,
                    help='name of the checkpoint to forge. Required for single-divergence mode.')

args = parser.parse_args()


def main():
    if args.mode == 'forging':
        n_runs = args.runs
        ckpt_dir = args.ckpt_dir
        out_dir = args.out_dir
        device = args.device
        dataset = args.dataset
        arch = args.arch
        batch_size = args.batch_size
        torch.manual_seed(42)
        g = torch.Generator()
        g.manual_seed(0)

        g_forging = torch.Generator()
        g_forging.manual_seed(1)
        helpers.forging(
            runs=n_runs,
            ckpt_dir=ckpt_dir,
            out_dir=out_dir,
            arch=arch,
            dataset=dataset,
            device=device,
            generator_benign=g,
            generator_forging=g_forging,
            M=args.num_forging_candidates,
            batch_size=batch_size,
        )

    elif args.mode == 'divergence':
        n_runs = args.runs
        ckpt_dir = args.ckpt_dir
        device = args.device
        dataset = args.dataset
        ckpt_list = helpers.select_random_checkpoints(
            dataset, args.arch, ckpt_dir, n_runs)

        for i, ckpt_name in enumerate(ckpt_list):
            start = time.time()
            torch.manual_seed(42)
            g = torch.Generator()
            g.manual_seed(0)
            helpers.forged_run(ckpt_dir, args.out_dir, args.forgery_dir, args.arch, dataset, ckpt_name,
                               args.num_epochs, args.batch_size, device, g, args.norm)
            end = time.time()
            print(f'Divergence took: {end - start}s')

    elif args.mode == 'single-divergence':
        ckpt_dir = args.ckpt_dir
        device = args.device
        dataset = args.dataset
        ckpt_name = args.ckpt_name

        start = time.time()
        torch.manual_seed(42)
        g = torch.Generator()
        g.manual_seed(0)
        helpers.forged_run(ckpt_dir, args.out_dir, args.forgery_dir, args.arch, dataset, ckpt_name,
                           args.num_epochs, args.batch_size, device, g, args.norm)
        end = time.time()
        print(f'Divergence for {ckpt_name} took: {end - start}s')

    elif args.mode == 'shuffle':
        dataloader = train_data.LoadData(
            args.dataset, True, args.batch_size, g)
        for i in range(20):
            epoch = rng.integers(1, 6)
            ts = rng.integers(0, len(dataloader.train_loader))
            x, linf_max, linf_min = helpers.non_commutatative_addition(
                args.arch, args.dataset, args.batch_size, args.num_shuffles, epoch+1, ts, g, rng, args.device)
            with open(os.path.join(args.ckpt_dir, f'../{args.dataset}_{args.arch}_shuffle.txt'), 'a') as f:
                f.write(
                    f'epoch: {epoch} ts: {ts} unique_sums: {x} linf_min: {linf_min} linf_max: {linf_max}\n')

    elif args.mode == 'shuffle_divergence':
        epoch = args.epoch
        ts = args.ts
        shuffles = 2
        linf = helpers.shuffle_divergence(
            args.arch, args.dataset, args.batch_size, args.device, shuffles, epoch, ts, g,
            g_forging, rng)
        os.makedirs(os.path.join(args.ckpt_dir, '../shuffle_divergence'))
        np.save(
            os.path.join(
                args.ckpt_dir,
                (
                    f'../shuffle_divergence/{args.dataset}_{args.arch}-shuffles_{shuffles}-'
                    f'epoch_{epoch}-ts_{ts}_shuffle_divergence.npy')
            ),
            linf)
    elif args.mode == 'plot':
        dir = args.ckpt_dir
        helpers.plot_data(dir, args.dataset, args.arch, [64], args.norm, args.training_steps)
    elif args.mode == 'plot_common':
        dir = args.ckpt_dir
        helpers.plot_data_common(dir, args.norm, args.training_steps)
    else:
        raise ValueError(f'Unknown mode: {args.mode}')


if __name__ == '__main__':
    main()
