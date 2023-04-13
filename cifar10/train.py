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
num_epochs = 10
batch_size = 1
test_size = 10000
n_targets = 10

class GD:
    def __init__(self, lr=1e-2, step_size = 1e-2):
        self.lr = lr
        self.step_size = step_size
    def step(self, params, grads):
        return [(-self.step_size * dw, -self.step_size * db) for(dw, db) in grads]
'''class ADAM:
    def __init__(self, step_size = 1e-2, beta1 = 0.1, beta2 = 0.1, network = None, loss = None, data_loader = None):
        self.net = network
        self.loss = loss
        self.dataloader = data_loader
        self.params = self.net.params
        self.m = 0
        self.v = 0
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.stepsize = step_size
    def update(self):
        self.t += 1
        for x, y in self.dataloader:
            gt = grad(self.loss)(self.params, x, y)
            mt = self.beta1 * self.m + (1-self.beta1) * gt
            vt = self.beta2 * self.v +(1-self.beta2) * jnp.dot(gt, gt)
            mt = mt/(1 - self.beta1**self.t)
            vt = vt/(1 - self.beta2**self.t)
            self.params = self.params - self.stepsize * mt/(1e-8 + jnp.sqrt(jnp.dot(vt, vt)))
'''

class Loss_function:
    def __init__(self, net):
        self.net = net
    def loss(self,params, x, y):
        return None
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
        preds = self.net.forward(params, x)
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
def train(net, optimizer, loss, Dataloader):
    def one_hot(x, k, dtype=jnp.float32):
        """Create a one-hot encoding of x of size k."""
        return jnp.array(x[:, None] == jnp.arange(k), dtype)
    loss2 = Single_loss(net)
    for x, y in Dataloader:
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
import networks
from dataloader import *

net = networks.MLP(layer_sizes)

if __name__ == '__main__':
    cifar10_dataset = CIFAR10('./datasets', download=True, transform=FlattenAndCast())
    training_generator = NumpyLoader(cifar10_dataset, batch_size=batch_size, num_workers=0)
    cifar10_testset = CIFAR10('./datasets', download=True, transform=FlattenAndCast(), train=False)
    test_generator = NumpyLoader(cifar10_testset, batch_size=test_size, num_workers=0)
    loss1 = Loss_func(net)
    #optimizer = ADAM(network=net, loss=loss1, data_loader=training_generator)
    for epoch in range(num_epochs):
        start_time = time.time()
        optimizer = GD()
        train(net, optimizer, loss1, training_generator)
        #optimizer.update()
        test_accuracy = test(net, test_generator)
        epoch_time = time.time() - start_time
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Test set accuracy {}".format(test_accuracy))
    print('Finish!')
