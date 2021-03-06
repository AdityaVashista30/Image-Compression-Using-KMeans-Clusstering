
# ### Task 1: Importing Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
plt.style.use("ggplot")

from skimage import io
from sklearn.cluster import KMeans

plt.rcParams['figure.figsize'] = (20, 12)

# ###  K-means Image Compression Python Function
def color_compression(img,k):
    input_img=io.imread(img)
    img_data=(input_img/255.0).reshape(-1,3)
    kmeans=MiniBatchKMeans(k).fit(img_data)
    k_colors=kmeans.cluster_centers_[kmeans.predict(img_data)]
    
    k_img=np.reshape(k_colors,(input_img.shape))
    fig,(ax1,ax2)=plt.subplots(1,2)
    fig.suptitle('K-Means Image Compression',fontsize=20)
    
    ax1.set_title('Compressed')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(k_img)
    
    ax2.set_title('Original')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(input_img)
    
    plt.subplots_adjust(top=0.9)
    plt.show()
    
"""
img=input("Enter image location: ")
k=int(input("Enter of distinct numbers you want in compressed image;0<k<256: " ))
color_compression(img,k) """
