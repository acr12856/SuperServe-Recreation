# MiniSuperNet: A Simplified SuperServe-Like Dynamic Inference Model

This project is a PyTorch-based prototype that reimplements core ideas from the research paper [**SuperServe: Fine-Grained Inference Serving for Unpredictable Workloads**](https://arxiv.org/abs/2312.16733), authored by Alind Khare, Dhruv Garg, Sukrit Kalra, Snigdha Grandhi, Ion Stoica, and Alexey Tumanov.

The goal of this is to explore latency-aware inference ideas that are explored in this paper, including **LayerSelect**, **WeightSlice**, and **SubnetNorm**.

TODO: This is a work in progress. I currently have a basic implementation in with basic implementations of the above main features as well as some benchmarking, however the accuracies of different subnets are not representative of what they theoretically should be (next up!).
---

## LayerSelect 
LayerSelect enables selective activation of layers during inference through a configurable `depth` parameter. The idea is that within subnets that exist within a SuperNet, we can gain
even more control by toggling layers. My implementation does this via residual blocks for a ResNet model. 

## WeightSlice 
WeightSlice enables dynamic control over the number of channels used in convolution layers via a `width_mult` parameter, that represents the % of available input/output
channels to use, as well as an `expand_ratio` parameter, that represents the ratio of output to input channels for every convolutional or fully connected layer. My implementation
only utilizes the `width_mult` parameter, but I will soon be adding `expand_ratio` which can add more width in hidden layers. 

## SubnetNorm
SubnetNorm is a essentially a normalization layer that stores precomputed `(mean, variance)` statistics for every subnet configuration (including those with different layers turned off via LayerSelect, etc).
These precomputed statistics are ideally generated through training. I currently have a dummy function as a placeholder, but will soon be adding a more sophisticated version to replicate these statistics. 

---

## Example Usage 
You can run the cells in the notebook, and the last cell will show some benchmarking information including latencies and accuracies. 
**NOTE** The accuracies are currently incorrect and are a problem I am working to fix.
