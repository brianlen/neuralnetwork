import numpy as np
import scipy

################# Neural_Network #################
class Neural_Network(object):
    def __init__(self):
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #Random initial weights
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        #Forward propagation
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self,z):
        #Activation function (activity layer)
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self,z):
        #Differentiation of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self,X,y):
        #Back propagation of error to weightsk
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self,X,y):
        #partial wrt W1 and W2
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def getParams(self):
        #Get W1 and W2 in one vector
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self,params):
        #Set W1 and W2 using single parameter vector
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

################# trainer #################

class trainer(object):
    def __init__(self,N):
        #Make local reference to Neural_Network:
        self.N = N

    def costFunctionWrapper(self,params,X,y):
        self.N.setParams(params)
        cost = self.N.costFunction(X,y)
        grad = self.N.computeGradients(X,y)
        return cost, grad

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def train(self,X,y):
        #make internal variable for callback function
        self.X = X
        self.y = y

        self.J = [] #empty list for storage
        params0 = self.N.getParams()
        options = {'maxiter':200, 'disp':True}

        #BFGS
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, \
            method='BFGS', args=(X,y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res


################# Initialize #################

#Supervised inputs
X = np.array(([3,5],[5,1],[10,2]), dtype=float)
y = np.array(([75],[82],[93]), dtype=float)

#Normalized
X = X/np.amax(X, axis=0)
y = y/100 #max test score 100

############### Neural Network ###############
NN = Neural_Network()
yHat = NN.forward(X)


cost1 = NN.costFunction(X,y)

scalar = 3

#plus
dJdW1, dJdW2 = NN.costFunctionPrime(X,y)
NN.W1 = NN.W1 + scalar*dJdW1
NN.W2 = NN.W2 + scalar*dJdW2
cost2 = NN.costFunction(X,y)

#minus
dJdW1, dJdW2 = NN.costFunctionPrime(X,y)
NN.W1 = NN.W1 - scalar*dJdW1
NN.W2 = NN.W2 - scalar*dJdW2
cost3 = NN.costFunction(X,y)

print cost1, cost2, cost3




################# Training #################

T = trainer(NN)

T.train(X,y)

plot(T.J)
grid(1)
xlabel('Iterations')
ylabel('Cost')
