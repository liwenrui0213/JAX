import jax
from jax import numpy as jnp
from jax import random
from jax import vmap,grad
import torchvision
from torchvision.datasets import CIFAR10
import numpy as np
from torch.utils import data


class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))

cifar10_dataset = CIFAR10('./datasets', download=True, transform=FlattenAndCast())

print(cifar10_dataset)
print(cifar10_dataset[0])
print(cifar10_dataset[0][0])