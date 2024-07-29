import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from GMM import GaussianMM
from Utils import HUES, image_load

#input image from the folder
file_name = input('Enter Image name: ')
image_path = 'images/'+file_name
image = image_load(image_path)

#check as RGB image
try:
    image_height, image_width, image_channels = image.shape
    
    image_type = 0
    #create the vector as np array
    image_pixels = np.reshape(image, (-1, image_channels))

#using greyscale image
except ValueError as e:
    image_height, image_width = image.shape

    image_type = 1
    #create the vector as np array
    image_pixels = np.reshape(image, (-1, 1))

# Input number of classes
K_param = int(input('Input K: '))

#Apply K-Means for initial weights, covariance for GMM
k_means = KMeans(n_clusters=K_param)
labels = k_means.fit_predict(image_pixels)
cl_means = k_means.cluster_centers_

cl_weights = []
cl_covariance = []

for i in range(K_param):
    dt = np.array([image_pixels[j, :] for j in range(len(labels)) if labels[j] == i]).T
    cl_covariance.append(np.cov(dt))
    cl_weights.append(dt.shape[1] / float(len(labels)))

#Create a GMM object
gmm = GaussianMM(cl_means, cl_covariance, cl_weights, K_param, image_type)

#Apply EM Algorithm
logs = []
prev_log_likelihood = None

#setting max iterations as 1000
for i in range(1000):
    pos_prob, log_likelihood = gmm.expectation(image_pixels) # E-step
    gmm.maximization(image_pixels, pos_prob)   # M-step
    print(f"Iteration {i+1} - Log_Likelihood: {log_likelihood}")
    logs.append(log_likelihood)
    
    #difference of logs to be negligible
    if prev_log_likelihood != None and abs(log_likelihood - prev_log_likelihood) < 1e-10:
        break
    prev_log_likelihood = log_likelihood

#Show Result
pos_prob, log_likelihood = gmm.expectation(image_pixels)
map_pos_prob = np.reshape(pos_prob, (image_height, image_width, K_param))

#for RGB image
if image_type == 0:
    segmented_map = np.zeros((image_height, image_width, 3))

    for i in range(image_height):
        for j in range(image_width):
            highest_post_prob = np.argmax(map_pos_prob[i, j, :])
            segmented_map[i,j,:] = np.asarray(HUES[highest_post_prob]) / 255.0
    
    plt.imshow(segmented_map)

# for greyscale image
else:
    segmented_map = np.zeros((image_height, image_width))

    for i in range(image_height):
        for j in range(image_width):
            highest_post_prob = np.argmax(map_pos_prob[i, j, :])
            segmented_map[i,j] = np.arange(256)[highest_post_prob]
    
    plt.imshow(segmented_map,cmap='gray')

#Save the segmented image as a PNG file
# output = 'results/'+file_name+'.png'
# plt.savefig(output, bbox_inches='tight', pad_inches=0)
# plt.title('Grayscale Segmented Map')
# plt.axis('off')
plt.show()

#Plot the negative log likelihood
plt.plot(np.array(logs))
plt.title('Negative Log Likelihood')
plt.xlabel('Iterations')
plt.ylabel('Negative Log Likelihood')
plt.show()
