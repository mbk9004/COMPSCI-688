import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
from scipy.special import expit 
from sklearn.metrics import mean_squared_error

'''
Consider a 20 × 20 grid structured model where each node is connected to its
four grid neighbors (edge nodes will have fewer connections), and where we have a bias of bi = 0, and a
constant interaction strength of wij = ¯ w. Now, implement the Gibbs sampling algorithm. For each value of
¯ w ∈ {0, 0.2, 0.4, 0.6, 0.8, 1}, run a single chain Markov chain with random initialization for 100 iterations.
(Where, again, one iteration is a full pass over the entire grid.) Plot the final samples for each ¯ w value (code).
'''

def gibbs_samppling(b,w,iteration,dim,y):
    width,height = dim

    for i in range(iteration):
        for j in range(height):
            for k in range(width):
            #neibor
                p_jk = get_conditional_distribution(b,w,y,j,k,width,height)
                #print(p_jk)
                y[j,k] = np.random.choice([-1, 1], p=[1-p_jk, p_jk])
        
    return y

def gibbs_sampling_independent(b,w,iteration,dim,y):
    
    height,width = dim
    #y = np.random.choice([-1, 1], size=dim)
    samples = []
    for i in range(iteration):
        for j in range(height):
            for k in range(width):
            #neibor
                p_jk = get_conditional_distribution(b,w,y,j,k,width,height)
                #print(p_jk)
                y[j,k] = np.random.choice([-1, 1], p=[1-p_jk, p_jk])
        samples.append(y.mean())

    return np.array(samples)


def get_conditional_distribution(b,w,y,j,k,width,height):
    s = 0
    pair = create_pair(j,k,height,width)
    prod = np.sum([(w * y[k]) for k in pair])
    s = prod + b
    s = 2*s
    return expit(s)



def create_pair(i,j,height,width):
    pairs = list()
    if i > 0:
        pairs.append((i-1,j))
    if i < height-1:
        pairs.append((i+1,j))
    if j > 0:
        pairs.append((i,j-1))
    if j < width-1:
        pairs.append((i,j+1))
    return pairs

def get_img_to_array(img):
    image = Image.open(img)
    data = np.asarray(image)
    print(data)
    data = np.where(data < 128, -1, 1)
    return data


def gibbs_samppling_for_q7(b,w,iteration,dim,y,x):
    np.random.seed(1)
    height,width = dim
    samples = []
    for i in range(iteration):
        for j in range(height):
            for k in range(width):
            #neihbor
                p_jk = get_conditional_distribution_for_q7(b,w,y,j,k,width,height,x)
                #print(p_jk)
                y[j,k] = np.random.choice([-1, 1], p=[1-p_jk, p_jk])
        samples.append(y.copy())

    return np.array(samples)

def get_conditional_distribution_for_q7(b,w,y,j,k,width,height,x):
    s = 0
    b_1 = b*x[j,k]
    pair = create_pair(j,k,height,width)
    prod = np.sum([(w * y[k]) for k in pair])
    s = prod + b_1
    s = 2*s
    return expit(s)

if __name__ == "__main__":
   
    w_values = [0, 0.2, 0.4, 0.6, 0.8, 1]
   
    #
    '''
    for w in w_values:
        sample  = gibbs_samppling(0,w,100,(20,20),y)
        plt.imshow(sample, cmap='gray')

        plt.title(f'w = {w}')
        plt.show()
    '''

    for w in w_values:
        y = np.random.choice([-1, 1], size=((20,20)))
        samples = gibbs_sampling_independent(0,w,100,(20,20),y)
      
        plt.plot(samples,label='%s w' % w)

    plt.legend()  
    plt.show() 
    
    noisy = get_img_to_array('data\\im_noisy.png')
    clean = get_img_to_array('data\\im_clean.png')
    dim = (clean.shape[0],clean.shape[1])
    '''
    sam = gibbs_samppling_for_q7(0.4,0.4,100,dim,clean.copy(),noisy.copy())
    sampel = sam.mean(axis=0)
    print(sampel)
    plt.imshow(sampel, cmap='gray')
    plt.show()
    print(mean_squared_error(clean,sampel))
    '''
    '''
    b_values = [0.1,0.5,1,1.5,2]
    for w in [0.2,0.4,0.6,0.8,1.0]:
        for b in b_values:
            q8 = gibbs_samppling_for_q7(b,w,100,dim,clean.copy(),noisy.copy())
            sam = q8.mean(axis=0)
            print(mean_squared_error(clean,sam),b,w)
            plt.imshow(sam, cmap='gray')
            plt.show()
    '''
    '''
    q8 = gibbs_samppling_for_q7(1,1,100,dim,clean.copy(),noisy.copy())
    sam = q8.mean(axis=0)
    plt.imshow(sam, cmap='gray')
    plt.show()
    '''