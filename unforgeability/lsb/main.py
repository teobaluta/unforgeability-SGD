import lsb.load_and_grad as load_and_grad
import train.utils as train_utils
import train.data as train_data
import torch
import argparse
import random
import os
import numpy as np
import lsb.utils
import time
import struct

random.seed(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
g = torch.Generator()
g.manual_seed(0)
rng = np.random.default_rng(42)

g_forging = torch.Generator()
g_forging.manual_seed(1)

parser = argparse.ArgumentParser()

parser.add_argument('--ckpt_dir', type=str, help='directory where checkpoints are stored.')
parser.add_argument('--out_dir', type=str, help='directory where checkpoints are stored.')
parser.add_argument('--arch', type=train_utils.Architecture,
                    default=train_utils.Architecture.LeNet5,
                    help='Type of model architecture.')
parser.add_argument('--dataset', type=train_utils.Dataset, default=train_utils.Dataset.MNIST,
                    help='The name of the dataset.')
parser.add_argument('--batch_size', type=int, help='batch_size for forging')
parser.add_argument('--precision', type=int, help='precision for forging')
parser.add_argument('--device', type=str, help='which gpu to run on')
parser.add_argument('--mode', type=str, help='which experiment to run')
parser.add_argument('--epoch', type=int, default=0, help='The epoch of the checkpoint.')
parser.add_argument('--ts', type=int, default=500, help='The training step of the checkpoint.')
parser.add_argument('--M', type=int, default=400, help='The number of batches to use in the check.')

args = parser.parse_args()

def float64_to_bin(num):
    s = struct.pack('!d', num)
    b = ''.join(format(c, '08b') for c in s)
    return b

def write_to_lsb(
    dataset, arch, batch_size, epoch, ts, ckpt_dir, precision, out_dir, device, M
):
    data_loader = train_data.LoadData(dataset, True, batch_size, g_forging)
    model = train_utils.factory_model(
        arch, data_loader.input_channels, data_loader.input_features, 10
    )
    model.load_state_dict(
        torch.load(os.path.join(ckpt_dir, f'{dataset}_{arch}-ckpt-epoch_{epoch}-ts_{ts}.pt'),
                   map_location=device)[
            'model_state_dict'
        ]
    )
    model.to(device)
    num_batches = len(data_loader.train_loader)
    num_samples = num_batches * batch_size

    total_params, _ = train_utils.flatten_gradient_ids(model)
    all_zero_cols = np.array(object=[x for x in range(0, total_params)])

    lsb_total_time = 0
    grad_comp_total_time = 0

    ind_set = set()
    total_grads = np.empty((0, total_params))

    grad_comp_batched_start = time.time()
    for batch_no, (grads, ind_arr) in enumerate(
        load_and_grad.compute_per_sample_grads(
            model,
            data_loader,
            batch_size,
            [],
            num_samples,
            on_the_fly=True,
            device=device,
        )
    ):
        grad_comp_batched_end = time.time()
        grad_comp_batched_time = grad_comp_batched_end - grad_comp_batched_start
        grad_comp_total_time = grad_comp_total_time + grad_comp_batched_time
        print(f'Gradient batch #{batch_no} / {M} time: {grad_comp_batched_time}s')
        if batch_no == M:
            break
        delete_rows = []

        # start = time.time()
        # Remove repeating indices
        ind_list = ind_arr.tolist()
        for i, ind in enumerate(ind_list):
            if ind in ind_set:
                delete_rows.append(i)
            else:
                ind_set.add(ind)
        # end = time.time()
        # print(f'Ind set time: {end - start}s')

        grads = np.delete(grads, delete_rows, axis=0)
        if args.mode == 'np-rank' or args.mode == 'save_grads':
            total_grads = np.concatenate((total_grads, grads), axis=0)
        # print(f'\nThe shape of the grads is: {grads.shape}')
        if args.mode == 'lsb':
            lsb_start = time.time()
            os.makedirs(os.path.join(out_dir, 'lsb_txt'), exist_ok=True)
            non_zero_lsb_path = os.path.join(
                out_dir,
                f'lsb_txt/{dataset}_{arch}-epoch_{epoch}-ts_{ts}-p_{precision}-non_zeros_lsb.txt')
            lsb.utils.get_lsb_fast(grads, precision, non_zero_lsb_path, 'a')
            lsb_end = time.time()
            lsb_time = lsb_end - lsb_start
            lsb_total_time = lsb_total_time + lsb_time
        grad_comp_batched_start = time.time()
        continue

    if args.mode == 'np-rank':
        start_time = time.time()
        grads_rank = np.linalg.matrix_rank(total_grads)
        end_time = time.time()
        path = os.path.join(out_dir, f'rank-{dataset}_{arch}.txt')
        with open(path, 'a') as f:
            f.write(
                f'epoch: {epoch} ts: {ts} rank: {grads_rank} run_time: {end_time - start_time}s'
            )

    if args.mode == 'save_grads':
        path = os.path.join(out_dir, f'{dataset}/grads_txt/')
        # This transforms the gradients into their bitstrings and saves them to a file
        # total_grads = total_grads.transpose()
        # bin_array = np.vectorize(float64_to_bin)(total_grads)
        # print(total_grads.shape)
        # os.makedirs(path, exist_ok=True)
        # np.savetxt(
        #     os.path.join(
        #         path, f'{dataset}_{arch}-epoch_{epoch}-ts_{ts}-non_zeros_grads.txt'), bin_array, fmt='%s')
        os.makedirs(path, exist_ok=True)
        np.savetxt(
            os.path.join(
                path, f'{dataset}_{arch}-epoch_{epoch}-ts_{ts}-non_zeros_grads.txt', total_grads))

    if args.mode == 'lsb':
        path = os.path.join(out_dir, f'log-{dataset}_{arch}-epoch_{epoch}-ts_{ts}-p_{precision}.txt')
        with open(path, 'a') as f:
            f.write(
                f'lsb_total_time: {lsb_total_time} grad_comp_total_time: {grad_comp_total_time} all_zero_cols: {all_zero_cols.shape}'
            )


def main():
    dataset = args.dataset
    arch = args.arch

    write_to_lsb(
        dataset,
        arch,
        batch_size=args.batch_size,
        epoch=args.epoch,
        ts=args.ts,
        ckpt_dir=args.ckpt_dir,
        precision=args.precision,
        out_dir=args.out_dir,
        device=args.device,
        M=args.M
    )


if __name__ == '__main__':
    main()

