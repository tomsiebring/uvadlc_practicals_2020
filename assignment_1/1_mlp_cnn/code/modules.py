"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.
    
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
    
        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.
    
        Also, initialize gradients with zeros.
        """
        
        ########################
        # PUT YOUR CODE HERE  # 
        #######################
        self.params = {
          'weight' : np.random.normal(0, 0.0001, (out_features, in_features)),    # W is N x M, where N is output, M is input
          'bias' : np.zeros(out_features)     # b is 1 x N, B is S x N
        }
        self.grads = {
          'W' : np.zeros(shape=(out_features,in_features)),
          'b' : np.zeros(shape=(1,out_features))
        }
        self.X = None
        self.S = None
        
        ########################
        # END OF YOUR CODE    #
        #######################
    
    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.X = x
        self.S = x.shape[0]
        B = np.tile(self.params['bias'], reps=(self.S,1))
        out = self.X @ self.params['weight'].T + B
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads['weight'] = dout.T @ self.X
        self.grads['bias'] = np.ones(shape=(1,self.S)) @ dout
        dx = dout @ self.params['weight']
        ########################
        # END OF YOUR CODE    #
        #######################
        return dx



class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    
    def __init__(self):
        """
        Store intermediate variable Y
        """
        self.Y = None

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        out = np.exp(x - x.max())
        out = out / np.einsum('ij->i', out).reshape(out.shape[0], 1)
        self.Y = out
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        product = dout * self.Y
        dx = product - np.einsum('ij->i', product).reshape(product.shape[0], 1) * self.Y
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    
    def __init__(self):
        """
        Store intermediate variable Y
        """
        self.S = None

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
    
        TODO:
        Implement forward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.S = x.shape[0]
        out = -1 * np.sum(y.reshape(x.shape[1],1) * np.log(x)) / self.S # reshape in case y not column vector
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
    
        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx = -1 * np.tile(y, reps=(self.S,1)) / x / self.S
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return dx


class ELUModule(object):
    """
    ELU activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        h = np.vectorize(lambda x: x if x > 0 else np.exp(x) - 1)
        out = h(x)
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dhdx = np.vectorize(lambda x: 1 if x > 0 else np.exp(x))
        out = dhdx(x)
        ########################
        # END OF YOUR CODE    #
        #######################
        return dx

if __name__ == '__main__':
  M = 3
  N = 2
  S = 6
  C = 4
  # module = LinearModule(M,N)
  # x = np.ones(shape=(S,M))
  # y = np.ones(shape=(S,N))
  # print("grads", module.grads)
  # print("params", module.params)
  # print("forward", module.forward(x).shape, module.forward(x))
  # print("backward", module.backward(y).shape, module.backward(y))
  # module_sm = SoftMaxModule()
  # print(module_sm.forward(np.ones(shape=(S,C))))
  # print(module_sm.backward(np.ones(shape=(S,C))))
  # y = np.ones(shape=(S,N))
  z = np.exp(np.ones(shape=(S,N)))
  z[0,1] = -5
  # print(np.sum(y))
  # arr = np.array([1, 2, 3])
  # type(arr)
  # scale = lambda x: x * 3 if (x > 2) else x / 3
  # func = np.vectorize(lambda x: x * 3 if x > 2 else x / 3)
  # print(func(arr))
  module_ELU = ELUModule()
  print(module_ELU.forward(z))


