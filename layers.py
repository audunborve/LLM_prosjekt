import numpy as np
from utils import onehot

class Layer:

    """
    Base class for layers in the neural network with forward and backward pass.
    """
    def __init__(self):
        
        return

    def forward(self,inputs):
        raise NotImplementedError

    def backward(self,grad):
        raise NotImplementedError
    
    def step_gd(self,alpha):
        """
        Performs a gradient descent step given learning rate.
        Assumes that the layer has a parameter dictionary "params" on the form

        params = {
            'w1': {         
                'w': w,         The parameter matrix
                'd': d,         The gradient of loss wrt the parameter matrix
                },
            'w2': {....},
            
        }
        where each parameter has a key 'w' for weights and 'd' for gradients.
        """
        for param in self.params:
            self.params[param]['w'] -= alpha*self.params[param]['d']




class Attention(Layer):

    def __init__(self,d,k,init_scale = 0.1):
        self.wK = np.random.randn(k,d)*init_scale
        self.wQ = np.random.randn(k,d)*init_scale
        self.wO = np.random.randn(k,d)*init_scale
        self.wV = np.random.randn(k,d)*init_scale
        self.params = {
            "wK":{'wK':self.w, 'dK':np.zeros_like(self.wK)},
            "wQ":{'wQ':self.w, 'dQ':np.zeros_like(self.wQ)},
            "wO":{'wO':self.w, 'dO':np.zeros_like(self.wO)},
            "wV":{'wV':self.w, 'dV':np.zeros_like(self.wV)}}
        self.soft = Softmax()

        return


    def forward(self,x):
        self.x = x
        n = np.shape(x)[2]
        rhs = np.einsum('kd,bdn->bkn', self.params['wK']['wK'], x)
        lhs = np.einsum('bdn, kd->bnk', x, self.params['wQ']['wQ'])
        input = np.einsum('bnk,bkn->bnn', rhs, lhs)
        B = np.zeros(n,n)
        i1,i2 = np.tril_indices(n,-1)
        B[i1,i2] -= np.inf
        self.A = self.soft.forward(input + B)
        zA = np.einsum('bdn, nn-> bdn', x, self.A)
        self.WVzA = np.einsum('kd, bdn-> bkn', self.params['wV']['wV'], zA)
        WOTWVzA = np.einsum('kd, bkn-> bdn', self.params['wO']['wO'], self.WVzA)
        return x + WOTWVzA


    def backward(self,grad):
        gO = np.einsum('kd, bdn->bkn', self.params['wO']['wO'], grad)
        gOV = np.einsum('kd, bkn-> bdn', self.params['wV']['wV'], gO)
        gS = self.soft.backward(np.einsum('bdn, bdn-> bnn'), self.x, gOV)
        gOVAT = np.einsum('bkj, ij->bki', gOV, self.A)
        zgS = np.einsum('bdn, bnn->bdn', self.x, gS)
        WQzgS = np.einsum('kd, bdn->bkn', self.params['wQ']['wQ'], zgS)
        WKTWQzgS = np.einsum('kd, bkn->bdn', self.params['wQ']['wQ'], WQzgS)
        zgST = np.einsum('bdj, bij->bdi', self.x, gS)
        WKzgST = np.einsum('kd, bdn-> bkn', self.params['wK']['wK'], zgST)
        WQTWKzgST = np.einsum('kd, bkn-> bdn', self.params['wQ']['wQ'], WKzgST)

        self.params['wO']['dO'] = np.einsum('bkn, bdn -> bkn', self.WVzA, grad)
        gOAT = np.einsum('bkj, ij->bki', gO, self.A)
        gOATzT = np.einsum('bkn, bdn->bkd', gOAT, self.x)
        self.params['wV']['dV'] = gOATzT
        self.params['wK']['dK'] = np.einsum('bkn, bdn-> bkd', WQzgS, self.x)
        self.params['wQ']['dQ'] = np.einsum('bdn, bdn->bdd', WKzgST, self.x)

        return grad + gOVAT + WKTWQzgS + WQTWKzgST
    


class Softmax(Layer):

    def __init__(self):
        return

    
    def forward(self,x):
        self.P = np.exp(x - x.max(axis=0,keepdims=True))
        self.Q = np.sum(self.P,axis=0,keepdims=True) + 10e-8
        self.z = np.divide(self.P,self.Q)
        return self.z

    def backward(self,grad):
        S = np.divide(self.P, self.Q*self.Q + 10e-8)
        sum = np.sum(grad*S, axis=0, keepdims=True)
        return grad*self.z - sum*self.P



class CrossEntropy(Layer):

    def __init__(self):
        return

    def forward(self,Z,y):
        b, self.r = np.shape(y)
        m = np.shape(Z)[1]
        self.Y = onehot(y) #Size bxmxr
        self.Yhat = Z[:,:,-self.r:] #Size bxmxn->bxmxr
        ones = np.ones(m)
        p = np.einsum('m,bmr->br', ones, self.Yhat*self.Y)
        q = -np.log(p)
        return np.mean(q)


    def backward(self):
        return -1/self.r * np.divide(self.Y,self.Yhat+10e-8)
    


class LinearLayer(Layer):

    """
    Linear Layer
    """
    def __init__(self,input_size, output_size,init_scale = 0.1):
        """
        Constructor takes input size and output size of layer 
        and scale for the weights
        """

        #Initialize weights using a sample from the normal distribution
        #scaled with the init_scale
        self.w = np.random.randn(output_size,input_size)*init_scale
        self.params = {"w":{'w':self.w,
                            'd':np.zeros_like(self.w)}}
        

    def forward(self,x):
        """
        Computes the affine transformation of the forward pass
        Stores input for backwards pass and returns output y = Wx.

        x: input, array of shape (batch_size, input_size, n) = (b,d,n)
        y: output, array of shape (batch_size, output_size, n) = (b,o,n)
        """

        self.x = x
        
        #Return output of layer
        #y = w@x
        y = np.einsum('od,bdn->bon',self.params['w']['w'],x)
        return y
        
    def backward(self,grad):
        """
        Performs backward pass.

        grad: gradient of loss wrt output of layer, shape (batch_size, output_size, n) = (b,o,n)
        """

        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt weight w: 
        #dL/dw = (1/B)*sum_b^B (grad_b@x_b^T)
        self.params['w']['d'] = np.einsum('bon,bdn->od',grad,self.x)/b

        #Return gradient of loss wrt input of layer
        #dL/dw = w@grad.T
        return np.einsum('od,bon->bdn',self.params['w']['w'],grad)
    

class Relu(Layer):
    """
    Relu activation function
    """

    def __init__(self):
        return

    def relu(self,x):
        #relu(x) = max(0,x)
        return np.maximum(np.zeros(x.shape), x)

    def forward(self,x):
        
        #Store input for backwards pass
        self.x = x
        return self.relu(x)

    def backward(self,grad):

        #dL/dx = grad * relu'(x)
        return grad * np.where(self.x > 0, np.ones_like(self.x), np.zeros_like(self.x))



class EmbedPosition(Layer):
    def __init__(self,n_max,m,d,init_scale=1e-1):   

        """
        n_max: maximum length of input sequence
        m: number of items in the vocabulary / number of integers
        d: embedding dimension
        """

        #Initialize a linear layer for the embedding
        self.embed = LinearLayer(m,d,init_scale)
        #Initialize the position embedding matrix
        self.w = np.random.randn(d,n_max)*init_scale

        #Initialize the parameter dictionary for weight with key "Wp"
        self.params = {"Wp":{'w':self.w,'d':None}}

    def forward(self,X):

        """
        Input:
            X: one-hot encoded array of shape (b,m,n).

        Output:
            z_0: array of shape (b,d,n)

        embed.forward(X) maps (b,m,n) to (b,d,n). 
        Assigns a column of size d to each integer in the sequence
        and add positional embedding matrix (params['Wp']['w'][:,:n]) (b,d,n).

        Equivalent to 

        z_0 = W_E@X + W_P[:,:n]

        """

        #We assume that n < n_max
        n = X.shape[-1]
        z_0 = self.embed.forward(X) + self.params['Wp']['w'][:,:n]
        return z_0
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - None
        """

        
        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt positional embedding w:
        self.params['Wp']['d'] = np.zeros_like(self.w)
        self.params['Wp']['d'] += np.sum(grad,axis=0)/b

        #Use backwards pass of the linear layer
        self.embed.backward(grad)

        #This is always the final layer, so we return None
        return None
    
    def step_gd(self,step_size):

        #We need to call the step_gd method of the linear layer
        self.embed.step_gd(step_size)

        #And since we override step_gd(), we use super 
        #which calls the step_gd() of the base class
        #and does gd for the paramters in the params dict
        super().step_gd(step_size)




class FeedForward(Layer):


    def __init__(self,d, p,init_scale = 0.1):
        """
        Input:
            d: input dimension of first layer and output of second
            p: output dimension of first and input of second.

        """

        #first linear layer with input size d and output size p
        self.l1 = LinearLayer(d,p,init_scale)

        #We use the Relu activation function
        self.activation = Relu()

        #second linear layer with input size p and output size d
        self.l2 = LinearLayer(p,d,init_scale)


    def forward(self,x):
        """
        Input:
            - x of shape (b,d,n)
        Output:
            - shape (b,d,n)

        This is equivalent to
        y = x + W2.T@Relu(W1@x)

         (W1,W2 are p x d)
        """

        self.x = x

        return x + self.l2.forward(self.activation.forward(self.l1.forward(x)))
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - derivative of loss wrt input x. Shape (b,d,n)
        
        """

        #We use backward pass of the linear layers and activation.
        #Recall that the backward pass reverse the order of the layers. 
        grad_feed_forward = self.l1.backward(self.activation.backward(self.l2.backward(grad)))

        #Since forward pass is x + W2.T@Relu(W1@x)
        return grad + grad_feed_forward


    def step_gd(self,step_size):

        #Call the step_gd method of the linear layers
        self.l1.step_gd(step_size)
        self.l2.step_gd(step_size)