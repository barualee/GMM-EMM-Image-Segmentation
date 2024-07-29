import numpy as np
from scipy.stats import multivariate_normal

class GaussianMM:
    
    def __init__(self, cl_means, cl_covariance, cl_weights, K_param, image_type):

        self.cl_means = np.asarray(cl_means)
        self.cl_covariance = np.asarray(cl_covariance)
        self.cl_weights = np.asarray(cl_weights)
        self.K_param = K_param
        #0 for RGB, 1 for greyscale
        self.image_type = image_type


    #expectation step
    #this method calculates the probability of input data pt belonging to the cluster.
    def expectation(self, data):
        
        probs = []
        for i in range(self.K_param):
            mu = self.cl_means[i, :]
            if self.image_type == 0: 
                cov = self.cl_covariance[i, :, :]
            else:
                cov = self.cl_covariance[i]
            cl_weight = self.cl_weights[i]

            #get probability density function to get the probability.
            unnormalized_prob = cl_weight * multivariate_normal.pdf(data, mean=mu, cov=cov, allow_singular=True)
            probs.append(np.expand_dims(unnormalized_prob, -1))
        
        # probabilities for all clusters for the data pt.
        preds = np.concatenate(probs, axis=1)
        
        # log likelihood by adding all prob across all clusters and taking log.
        log_likelihood = np.sum(preds, axis=1)
        log_likelihood = np.sum(np.log(log_likelihood))

        #normalising the probabilities by dividng with the sum of all probacross the clusters.
        preds = preds / np.sum(preds, axis=1, keepdims=True)
        
        return np.asarray(preds), log_likelihood



    #maximization step
    #update parameters of GMM based on post. prob confidence, each iteration of EM.
    def maximization(self, data, probs):

        new_cl_covariance, new_cl_means, new_cl_weights = [], [], []
        sum_probs = np.sum(probs, axis=0)
        
        
        for i in range(self.K_param):
            
            #calculate new mean.
            new_mu = np.sum(np.expand_dims(probs[:, i], -1) * data, axis=0)
            new_mu /= sum_probs[i]
            new_cl_means.append(new_mu)

            #calculate new covariance.
            data_shifted = np.subtract(data, np.expand_dims(new_mu, 0))

            expanded = np.multiply(np.expand_dims(probs[:, i], -1), data_shifted)
            transpose = np.transpose(expanded)
            new_cov = np.matmul(transpose, data_shifted)
            new_cov /= sum_probs[i]
            
            new_cl_covariance.append(new_cov)

            #calculate new cluster weight.
            #sum of prob of cluster / sum of prob of all clusters.
            new_cl_weights.append(sum_probs[i] / np.sum(sum_probs))

        #update the GMM parameters.
        self.cl_covariance = np.asarray(new_cl_covariance)
        self.cl_means = np.asarray(new_cl_means)
        self.cl_weights = np.asarray(new_cl_weights)