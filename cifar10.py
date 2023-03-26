import jax
from jax import numpy as jnp
from jax import random
from jax import vmap,grad
import torchvision
from torchvision.datasets import CIFAR10
import numpy as np
from torch.utils import data
from jax.scipy.special import logsumexp


layer_sizes = [3*128, 64, 64, 10]
step_size = 0.01
num_epochs = 1
batch_size = 1
test_size = 10000
n_targets = 10
#net
class Net:
    def __init__(self, train=True):
        if train:
            self.params = self.random_init()
        else :
            self.params = self.load()
    def random_init(self):
        def random_layer_params(m, n, key, scale=1e-2):
            w_key, b_key = random.split(key)
            return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
        # Initialize all layers for a fully-connected neural network with sizes "sizes"
        def init_network_params(sizes, key):
            keys = random.split(key, len(sizes))
            return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
        return init_network_params(layer_sizes, random.PRNGKey(0))
    def save(self):
        pass
    def load(self):
        pass

    def forward(self, params, x):
        def relu(x):
            return jnp.maximum(0, x)
        activations = x
        for w, b in params[:-1]:
            outputs = jnp.dot(w, activations) + b
            activations = relu(outputs)
        final_w, final_b = params[-1]
        logits = jnp.dot(final_w, activations) + final_b
        return logits - logsumexp(logits)
    def batched_forward(self, params, y):
        return vmap(self.forward, in_axes=[None, 0])(params, y)
    def loss_func(self, params, x, y):
        preds = self.batched_forward(params, x)
        return -jnp.mean(preds * y)
    def update(self, x, y):
        loss = self.loss_func
        grads = grad(loss)(self.params, x, y)
        laplacian = self.laplacian(self.params, x, y)
        params = [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(self.params, grads)]
        self.params = params
        return laplacian
    def accuracy(self, x, y):
        #target_class = jnp.argmax(y, axis=1)
        predicted_class = jnp.argmax(self.batched_forward(self.params, x), axis=1)
        return jnp.mean(predicted_class == y)
    def train(self, dataloader):
        def one_hot(x, k, dtype=jnp.float32):
            """Create a one-hot encoding of x of size k."""
            return jnp.array(x[:, None] == jnp.arange(k), dtype)

        for x, y in dataloader:
            y = one_hot(y, n_targets)
            laplacian = self.update(x, y)
            print('laplacian = {}'.format(laplacian))
    def test(self, dataloader):
        for x, y in dataloader:
            return self.accuracy(x, y)
    def laplacian(self, params, x, y):
        def hessian(f):
            grad = jax.grad(f, argnums=[1,2])
            return jax.jacfwd(grad, argnums=[1,2])
        hessian = hessian(self.loss_func)(params, x, y)
        laplacian = jnp.trace(hessian)
        return laplacian


#dataset
def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)
class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))



import time
net = Net()
if __name__ == '__main__':
    cifar10_dataset = CIFAR10('./datasets', download=True, transform=FlattenAndCast())
    training_generator = NumpyLoader(cifar10_dataset, batch_size=batch_size, num_workers=0)
    cifar10_testset = CIFAR10('./datasets', download=True, transform=FlattenAndCast(), train=False)
    test_generator = NumpyLoader(cifar10_testset, batch_size=test_size, num_workers=0)
    for epoch in range(num_epochs):
        start_time = time.time()
        net.train(training_generator)
        test_accuracy = net.test(test_generator)
        epoch_time = time.time() - start_time
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Test set accuracy {}".format(test_accuracy))
    print('Finish!')
