"""Compute per-sample gradients from a checkpoint and different statistics on them."""
import numpy as np
import os
import time
from functorch import make_functional_with_buffers, vmap, grad
from typing import Any, Sequence, Tuple, Iterator
import torch
import train.data as train_data
import train.utils as utils
import torch.nn as nn


def vmap_per_sample_grads(
    model: nn.Module,
    data_loader: train_data.LoadData,
    debug: bool = False,
    device: str = 'cuda:1',
) -> Sequence[Any]:
    """Loads the model parameters and computes gradients on all samples.

    Args:
        arch (nn.Module): _description_
        dataset (utils.Dataset): _description_
        ckpt_path (str): _description_
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        Sequence[Any]: _description_
    """
    loss_fn = nn.CrossEntropyLoss()
    fmodel, params, buffers = make_functional_with_buffers(model)

    def compute_loss_stateless_model(params, buffers, sample, target):
        batch = sample.unsqueeze(0).to(device)
        targets = target.unsqueeze(0).to(device)

        predictions = fmodel(params, buffers, batch)
        loss = loss_fn(predictions, targets)

        return loss

    ft_compute_grad = grad(compute_loss_stateless_model)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    for i, (data, targets, ind) in enumerate(data_loader.train_loader):
        vmap_time = time.time()
        ft_per_sample_grads = ft_compute_sample_grad(
            params, buffers, data, targets)

        vmap_time = time.time() - vmap_time

        # TODO should probably move these to a test
        if debug:
            no_vmap = time.time()
            per_sample_grads = utils.compute_sample_grads(
                model, loss_fn, data, targets, batch_size=128
            )

            no_vmap = time.time() - no_vmap

            print(
                f"Computed per-sample gradient for batch {i, vmap_time, no_vmap}")

        yield ft_per_sample_grads, ind

    if debug:
        # we can double check that the results using functorch grad and vmap match the results of hand processing each one individually:
        for per_sample_grad, ft_per_sample_grad in zip(
            per_sample_grads, ft_per_sample_grads
        ):
            assert torch.allclose(
                per_sample_grad, ft_per_sample_grad, atol=3e-3, rtol=1e-5
            )


def compute_per_sample_grads(
    model: nn.Module,
    data_loader: train_data.LoadData,
    batch_size: int,
    gradient_ids: Sequence[Tuple[int, int]],
    num_samples: int = 50000,
    on_the_fly: bool = False,
    device: str = "cuda:1",
) -> Iterator[Sequence[Any]]:
    """Computes the per-sample gradients in `gradient_ids` for `num_samples` in the training dataset.
    Uses the `vmap_per_sample_grads` to do so in a batched way.

    Args:
        arch (utils.Architecture): _description_
        dataset (utils.Dataset): _description_
        ckpt_path (str): _description_
        batch_size (int): _description_
        gradient_ids (Sequence[Tuple[int, int]]): _description_
        num_samples (int): _description_

    Yields:
        Iterator[Sequence[Any]]: _description_
    """
    if not gradient_ids and on_the_fly:
        total_params, _ = utils.flatten_gradient_ids(model)
        grads = np.empty(shape=(batch_size, total_params), dtype=np.float64)
    elif not gradient_ids:
        total_params, _ = utils.flatten_gradient_ids(model)
        grads = np.empty(shape=(num_samples, total_params), dtype=np.float64)
    else:
        grads = np.empty(
            shape=(num_samples, len(gradient_ids)), dtype=np.float64)
    for batch_no, (per_sample_grads, ind) in enumerate(
        vmap_per_sample_grads(model, data_loader, device=device)
    ):
        if batch_no * batch_size >= num_samples:
            break
        if not gradient_ids and on_the_fly:
            prev_params = 0
            for _, grads_params_tensor in enumerate(per_sample_grads):
                grads_params = (
                    grads_params_tensor.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                    .reshape(batch_size, -1)
                )
                # print(f'layer params shape {grads_params.shape}')
                params_no = grads_params.shape[1]
                # print(f'non-zero grads {np.count_nonzero(grads_params)} / {grads_params.shape[0] * grads_params.shape[1]}')
                grads[:, prev_params: prev_params + params_no] = grads_params
                # print(f'grads[{batch_size * batch_no} : {end},{prev_params}:{prev_params+params_no}]')
                prev_params += params_no

        elif not gradient_ids:
            prev_params = 0
            for _, grads_params_tensor in enumerate(per_sample_grads):
                grads_params = (
                    grads_params_tensor.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                    .reshape(batch_size, -1)
                )
                # print(f'layer params shape {grads_params.shape}')
                params_no = grads_params.shape[1]
                if num_samples < batch_size * (batch_no + 1):
                    end = num_samples
                else:
                    end = batch_size * (batch_no + 1)

                # print(f'non-zero grads {np.count_nonzero(grads_params)} / {grads_params.shape[0] * grads_params.shape[1]}')
                grads[
                    batch_size * batch_no: end, prev_params: prev_params + params_no
                ] = grads_params
                # print(f'grads[{batch_size * batch_no} : {end},{prev_params}:{prev_params+params_no}]')
                prev_params += params_no
        else:
            for i, grad_id in enumerate(gradient_ids):
                layer, param_id = grad_id[0], grad_id[1]
                grads_params = (
                    per_sample_grads[layer].detach(
                    ).cpu().numpy().astype(np.float64)
                )
                # print(
                #     f'Layer {grad_id[0]} shape {grads_params.shape}')
                select_grad = grads_params[:,].reshape(batch_size, -1)
                select_grad = select_grad[:, param_id].reshape(-1, 1)
                if num_samples < batch_size * (batch_no + 1):
                    end = num_samples
                else:
                    end = batch_size * (batch_no + 1)
                # print(f'Batch {batch_no} for {grad_id} saved at '
                # f'[{batch_size * batch_no}:{end}][{i}:{i+1}]')
                grads[batch_size * batch_no: end, i: (i + 1)] = select_grad[
                    0: end - batch_size * batch_no,
                ]

        yield grads, ind
