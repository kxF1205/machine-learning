from collections import defaultdict

import numpy as np


class NaiveBayes:
    def __int__(self):
        # Contructor
        self.alpha=1.0
        self.classes=None
        self.prior = None
        self.mu = None
        self.s = None
        self.conditional_prob = None


    def cal_feature_prob(self,feature):
        values = np.unique(feature)
        total_num = float(len(feature))
        # a dictionary to store the probability
        value_prob = {}
        for v in values:
            value_prob[v] = ((np.sum(np.equal(feature,v))+self.alpha)/(total_num+len(values)*self.alpha))
        return value_prob

    def fit(self, X, y, y_bin, con_index=[],cat_index=[],alpha=0.5):
        # X is the training data' dataset N*D, and y is the labels N*2, the first column is the character and second column is numerical value

        #alpha is the smoothing ratio
        self.alpha = alpha
        print(y)
        self.classes = np.unique(y)
        # initialize the condition_prob for the conditional probability P(xj| y=ck) for each features at every class
        self.conditional_prob = {}
        # this list is to store all the prior p(c)
        self.prior = np.zeros(self.classes.shape[0])
        self.con_index = con_index
        self.cat_index = cat_index
        self.mu = np.zeros((self.classes.shape[0],len(con_index)))

        self.s = np.zeros((self.classes.shape[0],len(con_index)))
        sample_num = float(len(y))

        for c_index, c in enumerate(self.classes):
            # calculate the prior of each class c , and it's the parameter u
            self.prior[c_index] += np.sum(y == c) / float(X.shape[0])
            #c_num = np.sum(y==c)
            #self.prior[c_index] +=((c_num+self.alpha))/(sample_num+len(self.classes)*self.alpha)
            #the nonzero indice by row
            inds = np.nonzero(y_bin[:,c_index])[0]

            #print(X[inds,:][:,con_index])

            self.mu[c_index, :] = np.mean(X[inds,:][:,con_index], 0)
            self.s[c_index, :] = np.std(X[inds,:][:,con_index], axis=0,dtype=float)
            #this one is a dictionary for store the conditional_prob of each features with every class
            self.conditional_prob[c_index] = {}
            # Calculate total counts of all the categories of each class
            for cat_i in self.cat_index :# cats_i is the array of index of categorical features
                feature = X[inds,:][:,cat_i]
                #this dictionary is like{class{feature{f_value:prob}}}, c_index, cat_i, feature are the keys
                self.conditional_prob[c_index][cat_i]=self.cal_feature_prob(feature)
        #print(self.conditional_prob)
        return self

    #calculate the gaussianlikelihood for the test set
    def GaussianLikelihood(self, X):
        # X is the testing set N_test*D
        #print(self.mu.shape)
        #print(self.mu)
        #print(self.s)
        #print(self.s.shape)
        likelihood = (X[:,None,self.con_index]-self.mu[None,:,:])/self.s[None,:,:]
        likelihood = (likelihood**2)*(-0.5)
        likelihood += -np.log(self.s*np.sqrt(2*np.pi))
        log_likelihood = np.nansum(likelihood,2)

        return log_likelihood  # N_test*C
    #use key target_value to get the prob from a dictionary
    def get_xj_prob(self,values_prob,target_value):
        return values_prob.get(target_value)


    #calculate the multinomiallikelihood for the test set
    def MultiLikelihood(self,X):
        mul_likelihood = np.zeros((X.shape[0],len(self.classes))) # this one is a N_test*C array
        i = 0
        for x in X:
            #print(x)
            for c_index in range(len(self.classes)):
                feature_prob= self.conditional_prob[self.classes[c_index]]
                #print(feature_prob)
                for a_feature in self.cat_index:
                   prob=self.get_xj_prob(feature_prob[a_feature],x[a_feature])
                   if prob is not None:
                       mul_likelihood[i, c_index] += np.log(prob)

            i+=1
        return mul_likelihood



    def predict(self, X):
        # X is the test set
        # if the dataset has binary/cat features, we use Bernoulli/multi , if the features has continuous distribution(real) we use guassian
        if self.cat_index==[]:
            log_posterior = np.log(self.prior[None,:])+self.GaussianLikelihood(X)
        elif self.con_index==[]:
            log_posterior = np.log(self.prior[None,:]) + self.MultiLikelihood(X)
        else:
            log_posterior = np.log(self.prior[None,:])+self.MultiLikelihood(X)+self.GaussianLikelihood(X)
        return np.argmax(log_posterior, axis=1)



    def evaluate_acc(self, y, y_predict):
        m = y.shape[0]

        count_correct = 0
        print("now is the prediction")
        print(y_predict)
        for i in range(m):
            if (y[i] == y_predict[i]):
                count_correct += 1

        return count_correct * 100 / m
