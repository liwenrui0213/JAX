import jax
from jax import numpy as jnp
from jax import random
from jax import vmap,grad
from jax.scipy.special import logsumexp
from jax import lax

'''
JAX 与pytorch之间，一个重要的区别就是JAX更倾向于函数式编程。所谓函数式编程，
就是要求程序的主体部分是由“函数”这个概念组成的。一个函数有这样的性质：
    对函数输入input，返回output，同时不引起任何额外的变化。
与一个object相比，调用object的某些方法会改变object的某些attribute，对应到
你的代码中，最典型的就是params这个变量：他不应该归为MLP这个class的内部的一个
attribute，而应当放到class外面。

我推荐的写法中（与jax官方提供的库haiku类似），一个神经网络的class应该具有以
下几个标准method：

class StandardModule:
  def __init__(self, attribute1, attribute2....)
    该方法对应这个class的初始化方法，里面应当赋值这个class固定的一些参数，比如网络
    每层的宽度，网络参数正态初始化时每层的variance等等
  
  def init(self, rng_key, ...)
    该方法通过一些random key来生成网络参数，本质上类似于你所写的MLP.random_init；
    所不同的是，你不应当在class内部生成随机数，而应当在外部，通过给函数input来指定随
    机数。不然可以预见，你在需要初始化两组网络参数时，所产生的初始化是一模一样的，会产
    生bug
    
  def apply(self, params, input1, input2, ...)
    该方法通过输入input返回网络的输出，本质上类似于你写的forward方法

由于这三个方法是通用且必须的，你应该写一个抽象类NNbase，在其中定义后两种方法为
抽象方法，并让你所定义的MLP继承自NNbase来保证统一性。关于抽象类与抽象方法，可以参考
这篇博客：https://www.cnblogs.com/jiyou/p/14024324.html

除此以外，对于一些简单的，没有训练时需要变化的params的模块，你可以按照标准的写
法，亦可以用以下更为简略的写法来完成：

def function_povider(attribute1, attribute2,...):
  def function(input1, ...):
    SOME CODE RELY ON attribute1, attribute2 ...
  return function

这种写法利用的python的一些特性，通过一个函数返回另一个函数，来实现一些固定参数的赋值。
他写起来更加简单，当然，对不熟悉的人会带来额外的困难，但你需要熟悉这一套写法。

你接下来的目标是：
1. 理解抽象类的概念，定义抽象类NNbase
2. 使用我上面讲的第一个方法重写你的MLP class，要求继承自你所写的抽象类NNbase
3. 将另一个文件中的L_2 loss function扩展为L_n loss function, 并通过给定n=2返回L2 loss;
要求使用第二种方法来获得这一代码。
'''


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

