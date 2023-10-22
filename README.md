# SimulateBiggerBatchSize
Chen Liu (chen.liu.cl2482@yale.edu)

Please kindly **Star** [![Github Stars](https://img.shields.io/github/stars/ChenLiu-1996/SimulateBiggerBatchSize.svg?style=social&label=Stars)](https://github.com/ChenLiu-1996/SimulateBiggerBatchSize/) this repo for better reach if you find it useful.

## Contributions
We provide a simple code snippet on how to simulate a bigger batch size when it is not directly achievable by your GPU.

## Motivation and Methods
Many of use don't have the luxury of owning thousdands of gigantic GPUs --- not everyone is a project lead at NVIDIA or Google. In many cases, we want to train models at bigger batch sizes than we can afford with our devices. How can we do that?

Suppose the biggest batch size achievable by the device is `n`, how do we simulate a bigger batch size `N`?

Many of us knows the concept: accumulating the gradient from backpropagation multiple times before a single back propagation is equivalent to summing the gradients.

So what we do is simple, assuming `N` is divisible by `n`:

1. Pass the data and compute the loss, accumulate the gradient, but don't gradient backprop each batch.
2. Perform one gradient backprop every `N/n` batches.

Also, be careful with normalizing the loss. If the loss function is something that uses mean aggregation (i.e., almost all loss functions I use), we need to divide the per-batch loss by a factor of `N/n` during the gradient accumulation, or otherwise the behavior will be inconsistent with actually running on a bigger batch size.

To put it in code:

This is a normal update.
```
loss = loss_fn(...)

opt.zero_grad()
loss.backward()
opt.step()
```

This is our update.
```
loss = loss_fn(...) / int(N/n)

loss.backward()

if batch_idx % int(N/n) == (int(N/n) - 1):
    opt.step()
    opt.zero_grad()
```

## Details
This repository currently only contains a single file, as an example on how you would do this.

## Usage
To use, simply look at and go through the example script, and apply the logic in your own project. Don't forget to give us a **star** if you use it and find it helpful.

## Citation
To be added
