# Convolution in CUDA From First Principles
Implementing a convolution layer in CUDA, based on Pytorch nn.conv2D. 

## Performance
On a A30 GPU, a unoptimized 4000 by 4000 convolution kernel runs in about 0.5 seconds. On a Intel Xeon Gold CPU, a 4000 by 4000 convolution runs in about 1.5 seconds.

## Repository Structure
Code for convolution in cuda is under src, in which you'll find CPU implementations of convolution and a GPU implementation. Performance and correctness testing is under the 'Tests' directory, while scripts for running the tests is under the 'Scripts' directory.

## Blog: https://syedshazli.github.io/posts/posts/convolution/
