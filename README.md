# PrivFDM

The code is based on the public implementations of Differentially Private Latent Diffusion Models [(DP-LDMs)](https://github.com/SaiyueLyu/DP-LDM) and Latent Diffusion Models [(LDMs)](https://github.com/CompVis/latent-diffusion). We would like to sincerely thank the authors of the original implementation for their open-source contributions, which this project builds upon.

# Setting Up Your Enviroment:
This project uses Conda as its package management tool which can downloaded [here](https://docs.conda.io/en/latest/). Once installed, clone the repository. The remainder of this document will assume the project is stored in a directory called `PrivFDM`.

```sh
cd PrivFDM/
conda env create -f environment.yaml
conda activate PrivFDM
```

# Training Your Own Models

Once you have chosen a private dataset, there are two steps to training your own differentially private federated diffusion models. In each step, you will need to create a configuration file that specifies the hyperparameter of each model. Example config files can be found in `PrivFDM/configs/`. By default, the training is conducted across 6 clients in the federated setting. To ensure stable training, especially under federated setting, we recommend using GPUs with at least 48 GB of memory.

**Step 1: Autoencoder Pre-training**

**Important**: After pre-training the autoencoder, please make sure to configure the corresponding autoencoder path for each client in the config file.
```
CUDA_VISIBLE_DEVICES=0 python main_autoencoder.py --base <path to autoencoder yaml> -t --gpus 0,
```

**Step 2: Private Pre-training and Fine-tuning**

**Important:** Due to implementation constraints, this step can only be run on a single GPU, specified by the `--accelerator gpu` command line argument.
```
CUDA_VISIBLE_DEVICES=0 python main_federated.py \
    --base <path to fine-tune yaml> \
    -t \
    --gpus 0, \
    --accelerator gpu
```

# Sampling

To sample from class-conditional models (e.g. MNIST, Fashion-MNIST, CIFAR-10):
```
CUDA_VISIBLE_DEVICES=0 python sampling/cond_sampling_test.py \
    -y path/to/config.yaml \
    -ckpt path/to/checkpoint.ckpt \
    -c 0 1 2 3 4 5 6 7 8 9
```


# Evaluation

We evaulated our models using FID and Acc (%). Code for both is available in the repository. For both methods, first follow the section above to generate sufficiently many samples from your model. Please note that evaluation metrics need to be computed separately for each client, as they are not aggregated automatically in the current implementation.

## FID

First, compute Inception network statistics for the real dataset
```
CUDA_VISIBLE_DEVICES=0 python fid/compute_dataset_stats.py \
    --dataset ldm.data.mnist.MNISTTrain \
    --args size:32 \
    --output mnist_train_stats.npz
```

Next, compute the statistics for the generated samples:
```
CUDA_VISIBLE_DEVICES=0 python fid/compute_samples_stats.py \
    --samples conditional_mnist_samples.pt \
    --output mnist_samples_stats.npz
```

Finally, compute FID:
```
CUDA_VISIBLE_DEVICES=0 python fid/compute_fid.py \
    --path1 mnist_train_stats.npz \
    --path2 mnist_samples_stats.npz
```

## Downstream Classification Accuracy
For MNIST, to compute the accuracy, the command is :
```
CUDA_VISIBLE_DEVICES=0 python scripts/sampling_and_accuracy.py \
    --yaml path/to/config.yaml \
    --ckpt path/to/checkpoint.ckpt
```

# Implementation Comments

We build our code on top of the [DP-LDMs](https://github.com/SaiyueLyu/DP-LDM) and [LDMs](https://github.com/CompVis/latent-diffusion) repository. Thanks to the authors for open sourcing their code!
