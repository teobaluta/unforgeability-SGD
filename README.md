# Unforgeability in SGD

This repository is an implementation of the procedure and experiments in the CCS'23 paper [Unforgeability in Stochastic Gradient Descent](https://dl.acm.org/doi/10.1145/3576915.3623093).

The code has the following directory structure:
```
unforgeability/
├── lib/
│   ├── Makefile
│   └── rref.c
├── rank/
│   ├── include/
│   │   ├── gaussian.h
│   │   └── read-files.h
│   └── src/
│       ├── approx.cpp
│       ├── gaussian.cpp
│       ├── lenet.cpp
│       ├── read-files.cpp
│       └── resnet.cpp
├── lsb/
│   ├── __init__.py
│   ├── load_and_grad.py
│   ├── main.py
│   └── utils.py
├── train/
│   ├── data.py
│   ├── globals.py
│   ├── main.py
│   ├── models.py
│   └── utils.py
└── approx_forgery/
    ├── __init__.py
    ├── helpers.py
    └── main.py

```

Here's the directory structure of the results and data that would be stored upon running all the experiments. For all further sections we will refer to the root of this as RESULTDIR. For example the path to shuffle_divergence below will be `RESULTDIR/mnist/shuffle_divergence`

```
├── mnist
    ├── lenet5-batch_indices-batch_size_64
    ├── lenet5-batch_size_64
    ├── lenet5-l2_forged_benign
    │   └── batch_size_64
    ├── lenet5-linf_forged_benign
    │   └── batch_size_64
    ├── lenet5_divergence_error
    │   └── batch_size_64
    │       ├── l2_forging
    │       └── linf_forging
    ├── lenet5_divergence_error_extended
    │   └── batch_size_64
    │       ├── l2_forging
    │       └── linf_forging
    ├── lsb_logs
    ├── lsb_txt
    └── shuffle_divergence
```
## Setting up the environment
This can be done using either conda or venv.
### Conda
To setup conda run the following commands in the root of the project directory
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
conda env create -n [env_name] --file environment.yml
```

## Training the model
In the paper we use LeNet5 and ResNet-mini for our experiments, the model definitions can be found in train/models.py

To train the LeNet5 and ResNet-mini models, navigate to the train directory and run the `main.py` file. You can specify the model architecture, dataset, batch size, number of epochs, learning rate, and other parameters as command-line arguments.

Before running the training script, make sure to define the DATADIR global variable in `train/globals.py` to specify the directory where the training data is stored. Similarly, you can define the RESULTDIR global variable in `train/globals.py` to specify the directory where the checkpoints will be stored (if you choose to save them).
```
usage: main.py [-h]
               --arch {lenet5,resnet-mini}
               --dataset {mnist,cifar10}
               --device {cpu,gpu}
               --batch_size BATCH_SIZE
               --num_epochs NUM_EPOCHS
               --lr LR
               --save_ckpts SAVE_CKPTS
               --num_classes NUM_CLASSES

Train a deep neural network on the specified dataset.

arguments:
  -h, --help            show this help message and exit
  --arch {lenet5,resnet-mini}
                        Model architecture to use
  --dataset {mnist,cifar10}
                        Dataset to use for training 
  --device DEVICE    Device to use for training
  --batch_size BATCH_SIZE
                        Mini-batch size for training
  --num_epochs NUM_EPOCHS
                        Number of epochs to train for
  --lr LR               Learning rate for optimizer
  --save_ckpts SAVE_CKPTS
                        Whether to save checkpoints during training
                        Set 1 to save and 0 otherwise
  --num_classes NUM_CLASSES
                        Number of classes in the dataset
                        (10 for both MNIST and LeNet5)
```
Here's an example command to train the LeNet5 model on the MNIST dataset:

```
python -m train.main --device cuda:0 --arch lenet5 --dataset mnist --batch_size 64 --num_epochs 10 --lr 0.01 --save_ckpts 1 --num_classes 10
```

And here's an example command to train the ResNet-mini model on the CIFAR10 dataset:

```
python -m train.main --device cuda:0 --arch resnet_mini --dataset cifar10 --batch_size 64 --num_epochs 20 --lr 0.01 --save_ckpts 1 --num_classes 10
```

When you run the `main.py` command with the specified options, the script will create a directory inside RESULTDIR with the name of the dataset you specified. Inside this directory, two more directories will be created:

* **{arch}-batch_size_{batch_size}**: This directory will contain the checkpoints for the trained model. Each checkpoint is saved as a .pt file, and contains a dictionary with the structure: `{epoch: , model_state_dict: , optimizer_state_dict: , loss: }`. The name of each checkpoint file is of the format `mnist_lenet5-ckpt-epoch_0-ts_0.pt`
* **{arch}-batch_indices-batch_size_{batch_size}**: This directory will contains a numpy array `batch_ind[i][j]` that has the batch indices from the PyTorch Dataset that was used to make up the batch used at ith epoch and jth training step.

Here's the directory structure for the output:
```
RESULTDIR/
└── {dataset}/
    ├── {arch}-batch_size_{batch_size}/
    └── {arch}-batch_indices-batch_size_{batch_size}/

```

## Approximate Forgery Experiments
There are four modes in which approx_forgery.main can be called, `{forging, divergence, shuffle, shuffle_divergence, plot}`

### Forging
```
usage: python -m approx_forgery.main [-h]
                                     --runs RUNS
                                     --ckpt_dir CKPT_DIR
                                     --out_dir OUT_DIR
                                     --arch ARCH
                                     --dataset DATASET
                                     --device DEVICE
                                     --batch_size BATCH_SIZE
                                     --mode MODE

arguments:
  -h, --help            show this help message and exit
  --runs RUNS           Number of random checkpoints to run forgery on.
  --ckpt_dir CKPT_DIR   Checkpoint directory to the benign checkpoints from.
  --out_dir OUT_DIR     Directory to store the results. Default: None
  --arch ARCH           Architecture of the model {lenet5, resnet_mini}
  --dataset DATASET     Dataset to use {mnist, cifar10}
  --device DEVICE       Device to use for training.
  --batch_size BATCH_SIZE
                        Batch size for training. Default: 64
  --mode MODE           Mode to run in out of {forging}

```

Here's an example command to perform approximate foring on 25 randomly sampled checkpoints from a training run of LeNet5 on MNIST with a batch size of 64.

```
python -m approx_forgery.main --runs 25 --ckpt_dir RESULTDIR/mnist/lenet5-batch_size_64/ --out_dir RESULTDIR/mnist/ --arch lenet5 --dataset mnist --device cuda:0 --batch_size 64 --mode forging
```
The execution of this command creates two directories containing forged checkpoints `RESULTDIR/mnist/lenet5-l2_forged_benign` and `RESULTDIR/mnist/lenet5-linf_forged_benign` using l2 and linf forging respectively. It also stores two numpy files `RESULTDIR/mnist/l2_error-lenet5-batch_size_64.npy` and `RESULTDIR/mnist/linf_error-lenet5-batch_size_64.npy` that contain the l2 and linf errors after forging for the 25 checkpoints.


### Divergence
By setting the mode to divergence you can reproduce the experiments that show how approximate forgeries diverge from the original trace as the model is trained for more training steps.

```
usage: python -m approx_forgery.main [-h]
                                     --runs RUNS
                                     --ckpt_dir CKPT_DIR
                                     --out_dir OUT_DIR
                                     --arch ARCH
                                     --dataset DATASET
                                     --device DEVICE
                                     --batch_size BATCH_SIZE
                                     --num_epochs NUM_EPOCHS
                                     --norm NORM
                                     --mode MODE

arguments:
  -h, --help            show this help message and exit
  --runs RUNS           Number of random checkpoints to run forgery on.
  --ckpt_dir CKPT_DIR   Checkpoint directory to the benign checkpoints from.
  --out_dir OUT_DIR     Directory to store the results.
  --arch ARCH           Architecture of the model {lenet5, resnet_mini}
  --dataset DATASET     Dataset to use {mnist, cifar10}
  --device DEVICE       Device to use for training.
  --batch_size BATCH_SIZE
                        Batch size for training.
  --num_epochs NUM_EPOCHS
                        Number of epochs to track the divergence error for
  --norm NORM           Norm to calculate the divergence error {l2, linf}
  --mode MODE           Mode to run in out of {divergence}

```
To run in this mode, its necessary to have run in forging mode before and saving the forged checkpoints as well as using the same base RESULTDIR

Here's an example command to perform divergence testing on 25 randomly sampled l2 forged checkpoints on a trace of LeNet5 trained on MNIST using a batch size of 64. The divergence l2 and linf error is measured for 5 epochs.

```
python -m approx_forgery.main --runs 25 --ckpt_dir RESULTDIR/mnist/lenet5-batch_size_64/ --out_dir RESULTDIR/mnist/ --arch lenet5 --dataset mnist --device cuda:0 --batch_size 64 --num_epochs 5 --norm l2 --mode divergence
```

The execution of this command creates the following directories `RESULTDIR/mnist/lenet5_divergence_error/batch_size_64/l2_forging`. Inside this directory are saved numpy arrays containing the l2 and linf divergence errors for every training step contained in 5 epochs for the 25 randomly sampled checkpoints.

### Shuffle
This mode is to test the non commutativity of floating point addition.

```
usage: python -m approx_forgery.main [-h]
                                     --ckpt_dir CKPT_DIR
                                     --out_dir OUT_DIR
                                     --arch ARCH
                                     --dataset DATASET
                                     --device DEVICE
                                     --batch_size BATCH_SIZE
                                     --mode MODE

arguments:
  -h, --help            show this help message and exit
  --ckpt_dir CKPT_DIR   Checkpoint directory to the benign checkpoints from.
  --out_dir OUT_DIR     Directory to store the results.
  --arch ARCH           Architecture of the model {lenet5, resnet_mini}
  --dataset DATASET     Dataset to use {mnist, cifar10}
  --device DEVICE       Device to use for training.
  --batch_size BATCH_SIZE
                        Batch size for training.
  --mode MODE           Mode to run in out of {shuffle}

```

Here's an example command to perform the test for commutativeness of floating point addition on 20 randomly sampled checkpoints where the grads of each checkpoint are shuffled in 1000 different orders.

```
python -m approx_forgery.main --ckpt_dir RESULTDIR/mnist/lenet5-batch_size_64/ --out_dir RESULTDIR/mnist/ --arch lenet5 --dataset mnist --device cuda:0 --batch_size 1024 --mode shuffle --num_shuffles 1000
```

The execution of this command leads to the creation of `mnist_lenet5_shuffle.txt` that contains the number of unique sums out of the 1000 different shuffle orders that were created.

### Shuffle Divergence
This mode is to test that the errors due to non commumtativity of floating point addition get more and more pronounced as training progresses.

```
usage: python -m approx_forgery.main [-h]
                                     --ckpt_dir CKPT_DIR
                                     --arch ARCH
                                     --dataset DATASET
                                     --device DEVICE
                                     --batch_size BATCH_SIZE
                                     --mode MODE
                                     --epoch EPOCH
                                     --ts TS

arguments:
  -h, --help            show this help message and exit
  --ckpt_dir CKPT_DIR   Checkpoint directory to the benign checkpoints from.
  --arch ARCH           Architecture of the model {lenet5, resnet_mini}
  --dataset DATASET     Dataset to use {mnist, cifar10}
  --device DEVICE       Device to use for training.
  --batch_size BATCH_SIZE
                        Batch size for training.
  --mode MODE           Mode to run in out of {shuffle_divergence}
  --epoch EPOCH         Epoch value of the checkpoint
  --ts TS               Training step of the checkpoint

```

Here's an example command to run shuffle divergence on LeNet5 trained on MNIST with batch size 64 at epoch 0 training step 45.

```
python -m approx_forgery.main --ckpt_dir RESULTDIR/mnist/lenet5-batch_size_64/ --arch lenet5 --dataset mnist --device cuda:1 --batch_size 1024 --mode shuffle_divergence --epoch 0 --ts 45
```

### Plots
To recreate the plot in Figure 1, use the following command (optionally send in a gradient id `--grad_id`):

```
python -m stats.main --ckpt_path RESULTDIR/mnist/lenet5/mnist_lenet5-ckpt-epoch_0-ts_500.pt --batch_size 10000
```

To recreate the plot in Figure 2, first ensure that you have run `approx_forger/main.py` in divergence mode. And that you have the divergence error data present in the directory `RESULTDIR/{dataset}/{arch}_divergence_error/batch_size_{batch_size}/{norm}_forging/`. You can then run the following command to generate the plot

```
python -m approx_forgery.main --ckpt_dir RESULTDIR --norm l2 --training_steps 3000 --mode plot_common
```

To recreate the plot in Figure 3, first ensure that you have run `approx_forger/main.py` in divergence mode. And that you have the divergence error data present in the directory `RESULTDIR/{dataset}/{arch}_divergence_error/batch_size_{batch_size}/{norm}_forging/` where you can replace the norm by the norm that you want to plot the errors for. You can then run the following command to generate the plot

```
python -m approx_forgery.main --ckpt_dir RESULTDIR --arch lenet5 --dataset mnist --norm linf --training_steps 3000 --mode plot
```
All the plots can be found at `../plots/` relative to the project directory.
Note: The plots require an installation of Tex on the system. You can do so by running :
```
sudo apt install texlive-latex-extra
```

## LSB file Creation
First create `librref.so` by running the following command in `lib/`.
```
make
```

Then to create the lsb.txt files that contain the LSBs computed with a fixed precision of the gradients at a particular checkpoint as a flattened out string use `lsb/main.py`.

```
usage: python -m lsb.main [-h]
                                     --ckpt_dir CKPT_DIR
                                     --out_dir OUT_DIR
                                     --arch ARCH
                                     --dataset DATASET
                                     --epoch EPOCH
                                     --ts TS
                                     --precision PRECISION
                                     --device DEVICE
                                     --batch_size BATCH_SIZE
                                     --mode MODE

arguments:
  -h, --help            show this help message and exit
  --ckpt_dir CKPT_DIR   Checkpoint directory to the benign checkpoints from.
  --out_dir OUT_DIR     Directory to store the results.
  --arch ARCH           Architecture of the model {lenet5, resnet_mini}
  --dataset DATASET     Dataset to use {mnist, cifar10}
  --epoch EPOCH         Epoch of the checkpoint
  --ts TS               Training step of the checkpoint
  --precision PRECISION Precision to use for calculating the LSB
  --device DEVICE       Device to use for training.
  --batch_size BATCH_SIZE
                        Batch size for training.
  --mode MODE           Mode to run in {lsb}

```

Here is an example command that computes the LSB

```
python -m lsb.main --ckpt_dir RESULTDIR/mnist/lenet5-batch_size_64/ --out_dir RESULTDIR/mnist/ --arch lenet5 --dataset mnist --epoch 0 --ts 100 --precision 26 --device cuda:0 --batch_size 64 --mode lsb
```

To create the gradients text files that are used to generate Table 4 for each checkpoint that you are interested in, you can run the following command:
```
python -m lsb.main --ckpt_dir RESULTDIR/mnist/lenet5-batch_size_64/ --out_dir RESULTDIR/ --arch lenet5 --dataset mnist --epoch 0 --ts 100 --precision 26 --device cuda:0 --batch_size 64 --mode save_grads
```
## Bool Rank Computation
To compute the rank to reproduce the results in the paper for LeNet5 and ResNet-mini, with the current directory as the project root run the following commands
```
cd rank
make
```
This should create three executables: `experiment-lenet`, `experiment-resnet`, `experiment-approx`

To run these executables you must provide as command line arguments the path where the required data is stored.
* **experiment-lenet**: For this you need to povide the path to the directory where the LeNet5 lsb text files are stored. If you generated them using `lsb/main.py` they should be in `RESULTDIR/mnist/lsb_txt/`. So the way to run the executable then would be:
```
./experiment-lenet RESULTDIR/mnist/lsb_txt/
```
This will generate a text file with the required results in the directory `rank/`

* **experiment-resnet**: For this you need to povide the path to the directory where the ResNet-mini lsb text files are stored. If you generated them using `lsb/main.py` they should be in `RESULTDIR/cifar10/lsb_txt/`. So the way to run the executable then would be:
```
./experiment-resnet RESULTDIR/cifar10/lsb_txt/
```
This will generate a text file with the required results in the directory `rank/`

* **experiment-approx**
For this you need to provide the path to the directory where the text files with the gradient values for the checkpoints you are interested in is stored. If you generated them using `lsb/main.py` they should be in `RESULTDIR/{dataset}/grads_txt/` So the way to run the executable then would be:
```
./experiment-approx RESULTDIR/{dataset}/grads_txt/
```
This will generate a text file with the required results in the directory `rank/`



## Accessing the Data

We make available our data on Amazon S3 bucket since the total size of the artifact is 1.1TB.

S3 storage: `https://artifact-unforgeability.s3.us-east-1.amazonaws.com/`

Each folder contains model checkpoints, files corresponding to the 25 checkpoints that contain the LSB, approximate forgery and floating-point divergence results. Each folder contains model checkpoints, files corresponding to the 25 checkpoints that contain the LSB, approximate forgery and floating-point divergence results.

## Maintainers

The code contributions are primarily done by Teodora ([teobaluta@gmail.com](mailto:teobaluta@gmail.com)), Racchit ([racchit.jain@gmail.com](mailto:racchit.jain@gmail.com)) and Ivica ([inikolic@nus.edu.sg](mailto:inikolic@nus.edu.sg)). Please feel free to reach out, and please cite our work if you are using our code or ideas.

