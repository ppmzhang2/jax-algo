# Jax-Algorithms

Computer vision deep learning algorithms implemented with
[JAX](https://jax.readthedocs.io/en/latest/) and
[Haiku](https://dm-haiku.readthedocs.io/en/latest/).

## Deploy

```sh
# normal deploy
make deploy
# deploy development environment
make deploy-dev
```

## MNIST Classification with ResNet

start training with the MNIST handwritten digit:

```sh
pdm run jaxalgo train-resnet --n-epoch=100
```
