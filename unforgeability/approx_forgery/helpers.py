"""Helper functions for `approx_forgery` module."""

import torch
import torch.nn as nn
import os
import random
import numpy as np
import train.data as train_data
import train.utils as utils
import random
import math
import matplotlib.pyplot as plt
import time
import tqdm
import lsb.load_and_grad as grad_help


# puts the weights into a list, but faster
def weights_to_list_fast(weights):
    with torch.no_grad():
        weights_list = []
        for weight in weights:
            list_t = weight.view(-1).tolist()
            weights_list = weights_list + list_t

        return weights_list


# set weight like above example, but faster
def set_weights_fast(x, weights):
    with torch.no_grad():
        start = 0
        # index = 0
        for weight in weights:
            length = len(weight.view(-1))
            array = x[start: start + length]
            weight_new = torch.Tensor(array).view(*weight.shape)

            weight.data.copy_(weight_new)
            # index +=1
            start += length


def select_random_checkpoints(dataset, arch, directory, k):
    # Get a list of all files in the directory
    all_files = os.listdir(directory)

    # Filter out any subdirectories in the list of files
    files = [f for f in all_files if os.path.isfile(
        os.path.join(directory, f))]

    # Choose k random files from the list that match the format
    # mnist_lenet5-ckpt-epoch_X-ts_Y.pt, where X is between 0 and 5
    matching_files = []
    for f in files:
        parts = f.split('-')
        if (
            len(parts) == 4
            and parts[0] == f'{dataset}_{arch}'
            and parts[1] == 'ckpt'
            and parts[2].startswith('epoch_')
        ):
            try:
                epoch_num = int(parts[2][6:])
                if 0 <= epoch_num <= 5:
                    matching_files.append(f)
            except ValueError:
                pass

    random_files = random.sample(matching_files, k)

    # Return the list of chosen file names
    return random_files


def get_batch(dataset, indices, generator):
    dataloader = train_data.LoadData(
        dataset, randomize='True', batch_size=64, generator=generator
    )
    dataset = dataloader.train_dataset
    images = []
    labels = []
    for ind in indices:
        img, label, _ = dataset[int(ind)]
        images.append(img)
        labels.append(label)
    return torch.stack(images), torch.tensor(labels)


def forged_run(
    ckpt_dir,
    out_dir,
    forgery_dir,
    arch,
    dataset,
    ckpt_name,
    num_epochs,
    batch_size,
    device,
    generator,
    norm,
):
    parts = ckpt_name.split('-')
    print(ckpt_name)
    epoch_str = parts[2].split('_')[1]
    ts_str = parts[3].split('_')[1].split('.')[0]
    epoch = int(epoch_str)
    ts = int(ts_str)

    dataloader = train_data.LoadData(dataset, True, batch_size, generator)
    model_forged = utils.factory_model(
        arch,
        dataloader.input_channels,
        dataloader.input_features,
        utils.get_num_classes(dataset),
    ).to(device)

    model_benign = utils.factory_model(
        arch,
        dataloader.input_channels,
        dataloader.input_features,
        utils.get_num_classes(dataset),
    ).to(device)

    forgery_ckpt_path = os.path.join(
            forgery_dir, f'{arch}-{norm}_forged_benign/batch_size_{batch_size}/',
            ckpt_name
            )
    st_forged = torch.load(
        forgery_ckpt_path,
        map_location=device
    )

    # Compare the forged checkpoint to the original checkpoint for the next
    # training step
    if ts < len(dataloader.train_loader) - 1:
        ts = ts + 1
    else:
        epoch = epoch + 1
        ts = 0

    st_benign = torch.load(
        os.path.join(ckpt_dir, f'{dataset}_{arch}-ckpt-epoch_{epoch}-ts_{ts}.pt'),
        map_location=device
    )['model_state_dict']

    model_benign.load_state_dict(st_benign)
    model_forged.load_state_dict(st_forged)

    loss_fn = nn.CrossEntropyLoss()
    optimizer_forged = torch.optim.SGD(model_forged.parameters(), lr=0.01)
    # DEBUG: Uncomment for retraining or debugging purposes
    # optimizer_benign = torch.optim.SGD(model_benign.parameters(), lr=0.01)
    divergence_error_l2 = []
    divergence_error_linf = []

    w_benign = np.concatenate([p.detach().cpu().flatten()
                              for p in st_benign.values()])
    w_forged = np.concatenate([p.detach().cpu().flatten()
                              for p in st_forged.values()])
    dif = w_forged - w_benign
    dif = np.abs(dif)
    linf = max(dif)
    l2 = math.sqrt(np.dot(dif, dif))
    divergence_error_l2.append(l2)
    divergence_error_linf.append(linf)
    print(f'[Before continuing training] L2 erorr: {l2}, Linf error: {linf}')

    batch_ind = np.load(
        os.path.join(
            ckpt_dir,
            f'../{arch}-batch_indices-batch_size_{batch_size}/{dataset}-{arch}-num_epochs_20-batch_indices.npy')
    )

    print(f'Running divergence from {epoch}, {ts} to {epoch + num_epochs}')
    times = []
    # Continue iterating through the training dataset, making sure the batches
    # are the same as the ones obtained when training the benign model
    for i in range(epoch + num_epochs):
        print(f'Epoch {i}')
        start_time = time.time()
        for j, (images, labels, indices) in tqdm.tqdm(enumerate(dataloader.train_loader)):
            assert(np.all(indices.numpy().astype(int) == batch_ind[i][j].astype(int)) == True)
            # Move the training dataset iterator to the ckpt epoch and ts
            if i < epoch:
                continue
            if i == epoch and j <= ts:
                continue

            images = images.to(device)
            labels = labels.to(device)
            out = model_forged(images)
            optimizer_forged.zero_grad()
            loss_forged = loss_fn(out, labels)
            loss_forged.backward()
            optimizer_forged.step()

            st_forged = model_forged.state_dict()
            w_forged = np.concatenate(
                [p.detach().cpu().flatten() for p in st_forged.values()]
            )

            # DEBUG or if we don't have the benign model saved:
            #   We retrain the benign model, but it
            #  takes longer than loading for larger models
            # out_benign = model_benign(images)
            # optimizer_benign.zero_grad()
            # loss_benign = loss_fn(out_benign, labels)
            # loss_benign.backward()
            # optimizer_benign.step()
            # st_benign_train = model_benign.state_dict()
            # w_benign_train = np.concatenate(
            #     [p.detach().cpu().flatten() for p in st_benign_train.values()]
            # )

            benign_model_path = os.path.join(ckpt_dir,
                                             f'{dataset}_{arch}-ckpt-epoch_{i}-ts_{j}.pt')
            st_benign = torch.load(benign_model_path, map_location=device)['model_state_dict']
            w_benign = np.concatenate(
                [p.detach().cpu().flatten() for p in st_benign.values()]
            )

            dif = w_forged - w_benign
            # DEBUG: This is for debugging purposes:
            #   It checks that the trained model is the same as the loaded one
            #   There might be differences if the training was done on a different
            #   machine even when the same seeds have been used.
            # dif_train = w_forged - w_benign_train
            # assert(np.all(w_benign == w_benign_train) == True)
            # assert(np.all(dif == dif_train) == True)

            dif = np.abs(dif)
            linf = max(dif)
            l2 = math.sqrt(np.dot(dif, dif))
            divergence_error_l2.append(l2)
            divergence_error_linf.append(linf)
        end_time = time.time()
        times.append(end_time - start_time)

    divergence_error_linf = np.array(divergence_error_linf)
    divergence_error_l2 = np.array(divergence_error_l2)

    path = (
        os.path.join(out_dir,
                     f'{arch}_divergence_error/batch_size_{batch_size}/{norm}_forging/')
    )
    os.makedirs(path, exist_ok=True)
    np.save(
        os.path.join(path,
                     f'l2_divergence_error-epoch_{epoch}-ts_{ts}.npy'),
        divergence_error_l2
    )
    np.save(
        os.path.join(path,
                     f'linf_divergence_error-epoch_{epoch}-ts_{ts}.npy'),
        divergence_error_linf
    )
    np.save(
        os.path.join(path,
                     f'divergence_error_times-epoch_{epoch}-ts_{ts}.npy'),
        np.array(times)
    )


def plot_data(
    parent_dir: str,
    dataset: utils.Dataset,
    arch: utils.Architecture,
    batch_sizes: list,
    norm: str,
    training_steps: int
):
    # plt.rcParams['text.usetex'] = True
    # create an empty list to hold the numpy arrays
    data = []

    # loop over all the directories and load the numpy arrays
    batch_sizes = [64]
    colors = ['blue']
    labels = ['Batch Size 64']
    # min_sizes = []
    for i, batch_size in enumerate(batch_sizes):
        directory = os.path.join(parent_dir, f'{dataset}/{arch}_divergence_error/batch_size_{batch_size}/{norm}_forging')
        batch_data = []

        for filename in os.listdir(directory):
            if filename.startswith(
                f'{norm}_divergence_error-epoch'
            ) and filename.endswith('.npy'):
                array = np.load(os.path.join(directory, filename))
                batch_data.append(array[0:training_steps])
                print(array.shape)

        data.append(batch_data)

    # compute the mean, max, and min of the data and plot them
    fig, ax = plt.subplots()
    for i, batch_size in enumerate(batch_sizes):
        batch_data = data[i]
        mean_data = np.mean(batch_data, axis=0)
        print(min(mean_data))
        print('\n')
        print(max(mean_data))
        max_data = np.max(batch_data, axis=0)
        min_data = np.min(batch_data, axis=0)
        ax.plot(range(training_steps), mean_data, label=labels[i], color=colors[i])
        ax.fill_between(range(training_steps), max_data, mean_data,
                        alpha=0.1, color=colors[i])
        ax.fill_between(range(training_steps), min_data, mean_data,
                        alpha=0.1, color=colors[i])

    ax.set_xlabel('Training steps')
    if norm == 'linf':
        ax.set_ylabel(r'$L_{\infty}$')
    else:
        ax.set_ylabel(r'$L_{2}$')

    os.makedirs('../plots', exist_ok=True)
    plt.savefig(
        f'../plots/forging-{dataset}-{arch}-batch_size_64-{norm}-divergence-{norm}.pdf')


def extract_epoch_ts(ckpt_name):
    parts = ckpt_name.split('-')
    epoch_str = parts[2].split('_')[1]
    ts_str = parts[3].split('_')[1].split('.')[0]
    epoch = int(epoch_str)
    ts = int(ts_str)
    return epoch, ts


def forging(
    runs,
    ckpt_dir,
    out_dir,
    arch,
    dataset,
    device,
    generator_benign,
    generator_forging,
    M,
    batch_size,
):
    start = time.time()
    data_loader = train_data.LoadData(
        dataset, randomize=True, batch_size=batch_size, generator=generator_benign
    )
    model1 = utils.factory_model(
        arch=arch,
        input_channels=data_loader.input_channels,
        input_features=data_loader.input_features,
        num_classes=utils.get_num_classes(dataset),
    )
    ckpt_list = select_random_checkpoints(dataset, arch, ckpt_dir, runs)
    results_l2 = []
    results_linf = []

    for i, ckpt_name in enumerate(ckpt_list):
        epoch, ts = extract_epoch_ts(ckpt_name)
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        # load the starting checkpoint of the benign run after which the forging step will happen
        st_benign_start = torch.load(ckpt_path)['model_state_dict']
        model1.load_state_dict(st_benign_start)
        model1 = model1.to(device)

        # load the final checkpoint of the bening run after the step where forging will happen
        if ts < len(data_loader.train_loader) - 1:
            final_ckpt_name = f'{dataset}_{arch}-ckpt-epoch_{epoch}-ts_{ts+1}.pt'
        else:
            final_ckpt_name = f'{dataset}_{arch}-ckpt-epoch_{epoch+1}-ts_{0}.pt'
        final_ckpt_path = os.path.join(ckpt_dir, final_ckpt_name)

        st_benign_final = torch.load(final_ckpt_path)['model_state_dict']
        model1.load_state_dict(st_benign_final)
        parameters_model1 = [parameter for name,
                             parameter in model1.named_parameters()]
        w_final = np.concatenate(
            [p.detach().cpu().flatten() for p in st_benign_final.values()]
        )

        data_loader_forgery = train_data.LoadData(
            dataset, randomize=True, batch_size=batch_size, generator=generator_forging
        )

        batch_ind = np.load(
            os.path.join(
                ckpt_dir,
                f'../{arch}-batch_indices-batch_size_{batch_size}/{dataset}-{arch}-num_epochs_20-batch_indices.npy'))

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model1.parameters(), lr=0.01)

        l2_min = 10000
        linf_min = 10000

        w_forged_list = []
        forgery_train_iterator = iter(data_loader_forgery.train_loader)
        print(f'Running forging for checkpoint ... {ckpt_name} [{i}/{len(ckpt_list)}]')
        for j in tqdm.tqdm(range(M)):
            model1.load_state_dict(st_benign_start)
            img, labels, ind = next(forgery_train_iterator)

            if len(data_loader_forgery.train_loader) - 1 == j:
                print('Not enough batches for forging. Reshuffling...')
                forgery_train_iterator = iter(data_loader_forgery.train_loader)

            if ts + 1 < len(data_loader_forgery.train_loader):
                benign_ind = batch_ind[epoch][ts + 1]
            else:
                benign_ind = batch_ind[epoch][0]
            if torch.equal(ind, torch.tensor(benign_ind, dtype=torch.int)):
                continue
            optimizer.zero_grad()
            img = img.to(device)
            labels = labels.to(device)
            out = model1(img)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
            st_forged = model1.state_dict()
            w_forged = np.concatenate(
                [p.detach().cpu().flatten() for p in st_forged.values()]
            )
            dif = w_final - w_forged
            dif_abs = np.abs(dif)
            l2 = np.sqrt(np.dot(dif_abs, dif_abs))
            linf = np.max(dif_abs)
            if l2 < l2_min:
                l2_min = l2
                l2_forged_weights = w_forged
            if linf < linf_min:
                linf_min = linf
                linf_forged_weights = w_forged
        print('forging done')

        l2_forged_weights = l2_forged_weights.tolist()
        linf_forged_weights = linf_forged_weights.tolist()
        with torch.no_grad():
            set_weights_fast(l2_forged_weights, parameters_model1)
            state_dict = model1.state_dict()
            path = os.path.join(out_dir,
                                f'{arch}-l2_forged_benign/batch_size_{batch_size}/')
            os.makedirs(path, exist_ok=True)
            torch.save(state_dict, os.path.join(path, ckpt_name))
        with torch.no_grad():
            set_weights_fast(linf_forged_weights, parameters_model1)
            state_dict = model1.state_dict()
            path = os.path.join(out_dir,
                f'{arch}-linf_forged_benign/batch_size_{batch_size}/')
            os.makedirs(path, exist_ok=True)
            torch.save(state_dict, os.path.join(path, ckpt_name))

        results_l2.append([epoch, ts, l2_min])
        results_linf.append([epoch, ts, linf_min])
    results_l2 = np.array(results_l2)
    results_linf = np.array(results_linf)
    np.save(
        os.path.join(out_dir, f'l2_error-{arch}-batch_size_{batch_size}.npy'),
        results_l2
    )
    np.save(
        os.path.join(out_dir, f'linf_error-{arch}-batch_size_{batch_size}.npy'),
        results_linf
    )
    end = time.time()
    print(f'Time taken for forging: {end-start}s')
    with open(os.path.join(out_dir, f'{arch}-batch_size_{batch_size}-M_{M}.txt'), 'w') as f:
        f.write(f'{end - start}\n')


def non_commutatative_addition(
    arch, dataset, batch_size, num_shuffles, num_epochs, ts, generator, rng, device
):
    data_loader = train_data.LoadData(dataset, True, batch_size, generator)
    model = utils.factory_model(
        arch, data_loader.input_channels, data_loader.input_features, 10
    )

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        for i, (img, label, ind) in enumerate(data_loader.train_loader):
            if epoch == (num_epochs - 1):
                if i == ts:
                    break
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            loss = loss_fn(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model = model.to(device)
    for grads, ind in grad_help.compute_per_sample_grads(
        model, data_loader, batch_size, [], batch_size, False, device
    ):
        break

    final_sum_vector = []
    for i in range(num_shuffles):
        rng.shuffle(grads)
        sum_vector = np.sum(grads, axis=0)
        final_sum_vector.append(sum_vector)
    unique_arrs = set(map(tuple, final_sum_vector))
    linf = []
    for i in range(len(final_sum_vector)):
        for j in range(i + 1, len(final_sum_vector)):
            linf_error = np.max(
                np.abs(final_sum_vector[i] - final_sum_vector[j]))
            linf.append(linf_error)
    print(len(linf))
    linf_max = max(linf)
    linf_min = min(linf)
    return len(unique_arrs), linf_max, linf_min


def shuffle_divergence(
    arch,
    dataset,
    batch_size,
    device,
    num_shuffles,
    num_epochs,
    ts,
    generator,
    generator_forging,
    rng,
):
    data_loader = train_data.LoadData(dataset, True, batch_size, generator)
    model = utils.factory_model(
        arch, data_loader.input_channels, data_loader.input_features, 10
    )

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        for i, (img, label, ind) in enumerate(data_loader.train_loader):
            if epoch == (num_epochs - 1):
                if i == ts:
                    break
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            loss = loss_fn(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model = model.to(device)
    for grads, _ in grad_help.compute_per_sample_grads(
        model, data_loader, batch_size, [], batch_size, device
    ):
        break
    print(grads.shape)
    final_sum_vector = []
    for i in range(num_shuffles):
        rng.shuffle(grads)
        sum_vector = np.sum(grads, axis=0)
        final_sum_vector.append(sum_vector)

    parameters_model = [parameter for name,
                        parameter in model.named_parameters()]

    w_start = weights_to_list_fast(parameters_model)
    final_weights = []
    model = model.to(device)
    for i in range(num_shuffles):
        run_weights = []
        data_loader_forging = train_data.LoadData(
            dataset, True, batch_size, generator_forging
        )
        w_start_np = np.array(w_start)
        run_weights.append(final_sum_vector[i])
        w_forged = w_start_np - 0.01 * (final_sum_vector[i])
        w_forged = list(w_forged)
        set_weights_fast(w_forged, parameters_model)
        for epoch in range(5):
            for j, (img, label, ind) in enumerate(data_loader_forging.train_loader):
                img = img.to(device)
                label = label.to(device)
                out = model(img)
                loss = loss_fn(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            w_end = weights_to_list_fast(parameters_model)
            w_end = np.array(w_end)
            run_weights.append(w_end)
        final_weights.append(run_weights)

    linf = []

    for i in range(len(final_weights[0])):
        linf_pairwise = []
        for j in range(len(final_weights)):
            for k in range(j + 1, len(final_weights)):
                linf_pairwise.append(
                    max(np.abs(final_weights[j][i] - final_weights[k][i]))
                )
        linf.append(linf_pairwise)

    linf = np.array(linf)
    return linf


def get_batch_data(directory, norm, training_steps):
    batch_data = []
    for filename in os.listdir(directory):
        if filename.startswith(f'{norm}_divergence_error-epoch') and filename.endswith(
            '.npy'
        ):
            array = np.load(os.path.join(directory, filename))
            batch_data.append(array[0:training_steps])
            print(array.shape)
    return batch_data


def plot_data_common(dir, norm, training_steps):
    # plt.rcParams['text.usetex'] = True
    # create an empty list to hold the numpy arrays

    # loop over all the directories and load the numpy arrays
    colors = ['blue', 'orange', 'red']
    arch_names = ['lenet5', 'resnet_mini']
    labels = ['LeNet5', 'ResNet-mini']
    directory_lenet5 = (
        os.path.join(dir,
                     f'mnist/lenet5_divergence_error/batch_size_64/{norm}_forging')
    )
    lenet_batch_data = get_batch_data(directory_lenet5, norm, training_steps)

    directory_resnet = (
        os.path.join(dir,
                     f'cifar10/resnet_mini_divergence_error/batch_size_64/{norm}_forging')
    )
    resnet_batch_data = get_batch_data(directory_resnet, norm, training_steps)

    # compute the mean, max, and min of the data and plot them
    fig, ax = plt.subplots()

    batch_data = [lenet_batch_data, resnet_batch_data]
    for batch_data, color, label in zip(batch_data, colors, labels):
        mean_data = np.mean(batch_data, axis=0)
        max_data = np.max(batch_data, axis=0)
        min_data = np.min(batch_data, axis=0)
        ax.plot(range(training_steps), mean_data, label=label, color=color)
        ax.fill_between(range(training_steps), max_data, mean_data, alpha=0.1, color=color)
        ax.fill_between(range(training_steps), min_data, mean_data, alpha=0.1, color=color)

    ax.set_xlabel('Training steps')
    if norm == 'linf':
        ax.set_ylabel(r'$L_{\infty}$')
    else:
        ax.set_ylabel(r'$L_{2}$')
    ax.legend()
    os.makedirs('../plots', exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        f'../plots/forging-lenet5-resnet-batch_size_64-{norm}-divergence-{norm}.pdf'
    )


def plot_vgg_divergence(dir, norm, training_steps):
    # plt.rcParams['text.usetex'] = True
    # create an empty list to hold the numpy arrays

    # loop over all the directories and load the numpy arrays
    colors = ['blue']
    arch_names = ['vgg_mini']
    labels = ['VGG-mini']
    directory_vgg = (
        os.path.join('/mnt/archive2/teodora/forgery-ccs-revision',
                     f'cifar10/vgg_mini-forging-M_400/vgg_mini_divergence_error/batch_size_64/{norm}_forging')
    )
    directory_vgg = '/mnt/archive2/teodora/forgery-ccs-revision/cifar10/vgg_mini-forging-M_400-extended/vgg_mini_divergence_error/batch_size_64/linf_forging/'
    vgg_batch_data = get_batch_data(directory_vgg, norm, training_steps)

    # compute the mean, max, and min of the data and plot them
    fig, ax = plt.subplots()

    batch_data = [vgg_batch_data]
    for batch_data, color, label in zip(batch_data, colors, labels):
        mean_data = np.mean(batch_data, axis=0)
        max_data = np.max(batch_data, axis=0)
        min_data = np.min(batch_data, axis=0)
        ax.plot(range(training_steps), mean_data, label=label, color=color)
        ax.fill_between(range(training_steps), max_data, mean_data, alpha=0.1, color=color)
        ax.fill_between(range(training_steps), min_data, mean_data, alpha=0.1, color=color)

    ax.set_xlabel('Training steps')
    if norm == 'linf':
        ax.set_ylabel(r'$L_{\infty}$')
    else:
        ax.set_ylabel(r'$L_{2}$')
    ax.legend()
    os.makedirs('../plots', exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        f'../plots/forging-vgg_mini-batch_size_64-{norm}-divergence_extended-{norm}.pdf'
    )