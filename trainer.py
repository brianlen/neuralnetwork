from videoSupport import *
from scipy import optimize

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
        #BFGS

        self.J = []
        params0 = self.N.getParams()
        options = {'maxiter':200, 'disp':True}

        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, \
            method='BFGS', args=(X,y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res
