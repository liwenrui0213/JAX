import jax
from jax import numpy as jnp
from jax import random
from jax import vmap,grad
import torchvision
from torchvision.datasets import CIFAR10
import numpy as np
from torch.utils import data
from jax.scipy.special import logsumexp


layer_sizes = [3*1024, 64, 64, 10]
num_epochs = 1
batch_size = 1
test_size = 10000
n_targets = 10
#net
class Net:
    def __init__(self, train=True, params_file=None):
        if train:
            self.params = self.random_init()
        else :
            self.params = self.load(params_file)
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
        def tanh(x):
            return jnp.tanh(x)
        activations = x
        for w, b in params[:-1]:
            outputs = jnp.dot(w, activations) + b
            activations = tanh(outputs)
        final_w, final_b = params[-1]
        logits = jnp.dot(final_w, activations) + final_b
        return logits - logsumexp(logits)
    def batched_forward(self, params, y):
        return vmap(self.forward, in_axes=[None, 0])(params, y)
#optmizer
class GD:
    def __init__(self, lr=1e-2, step_size= 1e-2):
        self.lr = lr
        self.step_size = step_size
    def step(self, params, grads):
        return -self.step_size * grads
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


class Loss_function:
    def __init__(self, net):
        self.net = net
    def loss(self,params, x, y):
        return 0
    def __call__(self, params, x, y):
        return self.loss(params, x, y)
class Loss_func(Loss_function):
    def __init__(self, net):
        super(self.__class__, self).__init__(net)
    def loss(self, params, x, y):
        preds = self.net.batched_forward(params, x)
        return -jnp.mean(preds * y)
class Single_loss(Loss_function):
    def __init__(self, net):
        super(self.__class__, self).__init__(net)
    def loss(self, params, x, y):
        preds = net.forward(params, x)
        return -jnp.dot(preds, y)
        #return jnp.dot(x, x)

def update(net, optimizer, loss, x, y):
    params = net.params
    grads = grad(loss)(params, x, y)
    delta = optimizer.step(params, grads)
    params = [(w + dw, b + db)
        for (w, b), (dw, db) in zip(params, delta)]
    net.params = params
def accuracy(net, x, y):
    #target_class = jnp.argmax(y, axis=1)
    params = net.params
    predicted_class = jnp.argmax(net.batched_forward(params, x), axis=1)
    return jnp.mean(predicted_class == y)
def train(net, optimizer, loss, dataloader):
    def one_hot(x, k, dtype=jnp.float32):
        """Create a one-hot encoding of x of size k."""
        return jnp.array(x[:, None] == jnp.arange(k), dtype)
    loss2 = Single_loss(net)
    for x, y in dataloader:
        y = one_hot(y, n_targets)
        update(net, optimizer, loss, x, y)
        laplacian = batched_laplacian(net, loss2, x, y)
        print('laplacian = {}'.format(laplacian))
def test(net, dataloader):
    for x, y in dataloader:
        return accuracy(net, x, y)

def batched_laplacian(net, f, x, y):
    def hessian(g):
        grad = jax.grad(g, argnums=1)
        return jax.jacfwd(grad, argnums=1)
    def laplacian(params, x, y):
        h = hessian(f)(params, x, y)
        return jnp.trace(h)
    params = net.params
    return vmap(laplacian, in_axes=[None, 0, 0])(params, x, y)


import time
net = Net()
if __name__ == '__main__':
    cifar10_dataset = CIFAR10('./datasets', download=True, transform=FlattenAndCast())
    training_generator = NumpyLoader(cifar10_dataset, batch_size=batch_size, num_workers=0)
    cifar10_testset = CIFAR10('./datasets', download=True, transform=FlattenAndCast(), train=False)
    test_generator = NumpyLoader(cifar10_testset, batch_size=test_size, num_workers=0)
    loss1 = Loss_func(net)
    for epoch in range(num_epochs):
        start_time = time.time()
        optimizer = GD()
        train(net, optimizer, loss1, training_generator)
        test_accuracy = test(net, test_generator)
        epoch_time = time.time() - start_time
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Test set accuracy {}".format(test_accuracy))
    print('Finish!')
