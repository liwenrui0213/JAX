import jax
from jax import numpy as jnp
from jax import random
from jax import vmap,grad
from jax.scipy.special import logsumexp

class MLP:
    def __init__(self, layer_sizes, train=True, params_file = None):
        if train:
            self.params = self.random_init()
            self.layer_sizes = layer_sizes
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


