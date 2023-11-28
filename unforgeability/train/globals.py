import torch


class MyGlobals(object):
    DATADIR = "/mnt/archive2/teodora/forgery-ccs-revision"
    RESULTDIR = "/mnt/archive2/teodora/forgery-ccs-revision"
    # save_grads_freq = 0

    # save_all_grads_ts = 5

    # Training args
    optimizer = "sgd"
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
    CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)
