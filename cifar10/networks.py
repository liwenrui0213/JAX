import jax
from jax import numpy as jnp
from jax import random
from jax import vmap,grad
from jax.scipy.special import logsumexp
from jax import lax

class MLP:
    def __init__(self, layer_sizes, train=True, params_file=None):
        self.layer_sizes = layer_sizes
        if train:
            self.params = self.random_init()
        else:
            self.params = self.load(params_file)
    def random_init(self):
        def random_layer_params(m, n, key, scale=1e-2):
            w_key, b_key = random.split(key)
            return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
        # Initialize all layers for a fully-connected neural network with sizes "sizes"
        def init_network_params(sizes, key):
            keys = random.split(key, len(sizes))
            return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
        return init_network_params(self.layer_sizes, random.PRNGKey(0))
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

class CONV:
    def __init__(self, size):
        self.size = size  #HWIO
        self.params = jnp.transpose(jnp.zeros(size, dtype=jnp.float32), [3, 2, 0, 1])  #transpose to OIHW
    def forward(self, params, x):   #x.shape = NCHW
        x = jnp.reshape(x, (1, *x.shape))     #CHW -> 1CHW
        kernel = params
        out = lax.conv(x, kernel, (1, 1), 'SAME')
        #def ReLU(x):
            #return jnp.max(0, x)
        def tanh(x):
            return jnp.tanh(x)
        out = tanh(out)
        out = jnp.squeeze(out, 0)
        return out
class Linear:
    def __init__(self, size):   #size=(3*1024, 10)
        self.size = size
        self.params = (jnp.zeros(size), jnp.zeros((size[1],)))
    def forward(self, params, x):  #x.shape = (3*1024)
        w = params[0]
        b = params[1]
        out = jnp.dot(x, w) + b
        def tanh(x):
            return jnp.tanh(x)
        out = tanh(out)
        return out

class CNN:
    def __init__(self, train=True, params_files=None):
        if train:
            self.conv1 = CONV((32, 32, 3, 3))
            self.conv2 = CONV((32, 32, 3, 3))
            self.conv3 = CONV((32, 32, 3, 3))
            self.Linear1 = Linear((3*1024, 10))
            self.params = [self.conv1.params, self.conv2.params, self.conv3.params, self.Linear1.params]

    def forward(self, params, x):
        out = self.conv1.forward(params[0], x)
        out = self.conv2.forward(params[1], out)
        out = self.conv3.forward(params[2], out)
        out = jnp.reshape(out, (3*1024))
        out = self.Linear1.forward(params[3], out)
        return out - logsumexp(out)
    def batched_forward(self, params, x):
        return vmap(self.forward, in_axes=[None, 0])(params, x)

