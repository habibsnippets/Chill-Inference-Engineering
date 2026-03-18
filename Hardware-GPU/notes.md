# GPU Architecture

designed for massive parallel workloads


## compute

gpus have SMs (streaming multiprocessors), each SM has multiple cores. three types of compute in GPUs :

* CUDA core : operates on individual numbers
* tensor core : operates on vectors and matrices
* SFU : accelrates certain math ops like sin cos etc

to measure inference compute , measure in terms of tensor core compute as they are responsible for MMA - matrix multiply and accumulate

compute is measured in flops

if see the spec sheet, you see 2 measurements of tensor compute :

* dense : raw flops if every element of the tensor is used
* sparse : in tensors with 2:4 structural sparsity where 50% of values are 0, tensor cores can skip mulitplication by 0.

FLOPS generally double with each halving of precision. A GPU capable of one petaFLOPS on 16-bit numbers will be able to do two petaFLOPS on 8-bit numbers.

## memory and caches

There are two types of memory on any chip, CPU or GPU:
• DRAM (Dynamic RAM): General-purpose off-chip memory denominated in gigabytes.
• SRAM (Static RAM): Faster, more expensive, on-chip memory denom inated in kilobytes or megabytes.

VRAM is a type of DRAM. V stands for Video.

