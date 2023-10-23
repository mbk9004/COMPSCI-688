import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
from scipy.optimize import minimize
import math
import time
import matplotlib.pyplot as plt  


'''
For the first test word only, compute the node potentials for each i
Fi

'''
def q1(x,mf):
    pt = mf.dot(x.T)
    return pt
def q2(x,mf,mp,k):
    energy_list =[]
    for i in range(len(x)):
        matrix = q1(x[i],mf)
        words = open('./data/test-words.txt', 'r').read().splitlines()
        current_word = list(words[i])
        y = [k[current_word[j]] for j in range(len(current_word))]
        
        log_pair = 0
        log_poten = 0
        for yi in range(len(y)):
            log_pair += matrix[int(y[yi])][yi]
        for yj in range(len(y)-1):
            log_poten += mp[int(y[yj])][int(y[yj+1])]
        energy_list.append(log_pair+log_poten)
    return energy_list
#summation over all possible character label
#sequences to compute the value of the log normalizing constant for the CRF model after conditioning on thecorresponding observed image sequence

    
def q3(x,mf,mp,key):
    for i in range(len(x)):

        all_lable =  all_possible_character_lable(len(x[i]))
       
        #value of the log normalizing constant for the CRF model
        lmc = 0.0
        for yi in all_lable:
             lmc += np.exp(energy(x[i], yi, mf, mp))
        print(np.log(lmc))
    return
def q4(x,mf,mp,key):
     for i in range(len(x)):
        all_lable =  all_possible_character_lable(len(x[i]))
        lmc = 0.0
        energy_list =[]
        for yi in all_lable:
             e = energy(x[i], yi, mf, mp)
             energy_list.append(e)
             lmc +=np.exp(e)
        max_energy = np.argmax(energy_list)
        log_lmc = np.log(lmc)
        p = np.exp(energy_list[max_energy]-log_lmc)
       
        word = []
        for i in (all_lable[max_energy]):
            word.append(key[i])

        print('label and probability')
        print(word,p)
def q5(x,fp,tp,key):
   
    all_lable =  all_possible_character_lable(len(x))
    lmc = 0.0
    #print(matrix)
    for yi in all_lable:
        lmc += np.exp(energy(x, yi, fp, tp))
 
   
    return_mat = np.zeros((len(key),len(x)))

    for i in range(len(key)):
        for j in range(len(x)):
            ms = all_possible_character_lable(len(x)-1)
            mat = []
            for m in ms:
                mat.append(list(m)[0:j]+[i]+list(m)[j:])
            seq = []
            for y in mat:
                seq.append(np.exp(energy(x, y, fp, tp)))
           
            return_mat[i][j] = np.sum(seq) / lmc
    print(return_mat)

    return
#For the first test word only, condition on the observed image sequence to obtain a chainstructured
#Markov network. Compute the log-space messagesm1->2(y2),m2!1(y1),m4!3(y3), andm3!2(y2)
#(code). Report the value of each message in a table
def q6(x,fp,tp,key2,m1,m2):
    model = construct_chain_dict(len(x)).copy()
    log_message = [] 
    passing_val = {}
    for y in range(len(key2)):
       
        log_message.append(get_pair(x,y,fp,tp,m1,m2,model,len(key2),passing_val))
    
    return log_message
def q7(x,fp,tp,key2):
    '''
    For the first test word only, use the computed messages to compute marginal probability
distributions (code). Report single variable marginals over each position in the word as a table. Represent
pairwise marginals over each adjacent node pairs as three tables, and report only the 3 3 block of entries
between the labels e; t; r in each table (report).
    '''
    model = construct_chain_dict(len(x)).copy()
    init = init_log(model,x,fp,tp,key2)
    return_mat = np.zeros((len(key2),len(x)))

    for i in range(len(key2)):
        for j in range(len(x)):
            return_mat[i][j] = single_variable_marginal(j,i,model,init,fp,tp,x,key2)
    print(return_mat)

    list_etr = [0,1,8]
   
    for etr in range(len(x)-1):
        mat = np.zeros((3,3)) #3by3
        for i in range(3):
            for j in range(3):
                mat[i][j] = pairwise_marginal_adjacent(model,etr,etr+1,list_etr[i],list_etr[j],x,fp,tp,key2)
        print (mat,etr+1,etr+2)
    return
def pairwise(x,fp,tp,key2):
    model = construct_chain_dict(len(x)).copy()
    list_etr = [0,1,2,3,4,5,6,7,8,9]
    mat = np.zeros((len(x)-1,10,10))
    for etr in range(len(x)-1):
        for i in range(10):
            for j in range(10):
                mat[etr][i][j] = pairwise_marginal_adjacent(model,etr,etr+1,list_etr[i],list_etr[j],x,fp,tp,key2)
        
    return mat
def q8(x,fp,tp,key2,key1):
    words = open('./data/test-words.txt', 'r').read().splitlines()
    all_char =0
    correct =0
    for k in range(len(x)):
        word = words[k]
        all_char+= len(x[k])
        model ={}
        model = construct_chain_dict(len(x[k])).copy()
        init = init_log(model,x[k],fp,tp,key2)
        return_mat = np.zeros((len(key2),len(x[k])))

        for i in range(len(key2)):
            for j in range(len(x[k])):
                return_mat[i][j] = single_variable_marginal(j,i,model,init,fp,tp,x[k],key2)
        
        max_pos = np.argmax(return_mat,0)
        
        predict_word =[]
        for i in (max_pos):
            predict_word.append(key1[i])
        if k <5:
            print (predict_word)
        
        label_word =[]
        for i in list(word):
            label_word.append(key2[i])
        correct += (sum(label_word==max_pos))
    print(correct/all_char)

def pairwise_marginal_adjacent(m,etr1,etr2,etr_i,etr_j,x,fp,tp,key2):
    init = init_log(m,x,fp,tp,key2)
    psi = x[etr1].dot(fp[etr_i]) + x[etr2].dot(fp[etr_j]) + tp[etr_i][etr_j]
    pro = 0
    for neighbor in m[etr1]:
        if neighbor!= etr2:
            passing = {}
            pro += get_pair(x,etr_i,fp,tp,neighbor,etr1,m,len(key2),passing)
    for neighbor in m[etr2]:
        if neighbor!= etr1:
            passing = {}
            pro += get_pair(x,etr_j,fp,tp,neighbor,etr2,m,len(key2),passing)
    return np.exp(psi + pro -init)


def single_variable_marginal(j,i,m,init,fp,tp,x,key):
    total = x[j].dot(fp[i])-init
    for neighbor in m[j]:
        passing ={}
        total += get_pair(x,i,fp,tp,neighbor,j,m,len(key),passing)
    
    return np.exp(total)

def init_log(model,x,fp,tp,key2):
        total =0
        
        for y in range(len(key2)):
            log_message=[]
            
            log_message.append(x[0].dot(fp[y]) ) 
            for t in model[0]:
                if int(t) != 0:
                    passing ={}
                    log_message.append(get_pair(x,y,fp,tp,int(t),0,model,len(key2),passing))
                
            total += np.exp(np.sum(log_message))

        return np.log(total)

def get_pair(x,y,fp,tp,fr,to,model,key_r,passing):
    val = 0
    if (fr,to,y) not in passing:
        for i in range(key_r):
            sum_si_fr_si_fr_to = x[fr].dot(fp[i]) +tp[y][i] 
            for neighbor in model[fr]:
                if neighbor != to:
                 
                   sum_si_fr_si_fr_to += get_pair(x,i,fp,tp,neighbor,fr,model,key_r,passing)
            val += np.exp(sum_si_fr_si_fr_to)

        passing[(fr,to,y)] = np.log(val)
    
    return passing[(fr,to,y)]

      

def construct_chain_dict(x):
    #return dic
    model_dict = {}
    for i in range(x):
        model_dict[i] = [i-1,i+1]
    model_dict[0] = [1]
    model_dict[x-1] = [x-2]
    return model_dict

def all_possible_character_lable(r):
    lable = list(it.product([0,1,2,3,4,5,6,7,8,9],repeat=r))
    return lable
def energy(x,y,mf,mp):
    matrix = q1(x,mf)
    log_pair = 0
    log_poten = 0
    for yi in range(len(y)):
        log_pair += matrix[int(y[yi])][yi]
    for yj in range(len(y)-1):
        log_poten += mp[int(y[yj])][int(y[yj+1])]
    return (log_pair+log_poten)   

     
def q9(x,fp,tp,key):
    words = open('./data/train-words.txt', 'r').read().splitlines()
    all_char =0
    liklihood =0
    for k in range(len(x)):
        word = words[k]
        all_char+= len(x[k])
        y =[]
        for i in list(word):
            y.append(key[i])
        model ={}
        model = construct_chain_dict(len(x[k])).copy()
        init = init_log(model,x[k],fp,tp,key2)
        liklihood += energy(x[k],y,fp,tp) - init
    print(liklihood/50.0)



def question13(x0,length):
    sums=0
    tp = x0[0:10*10].reshape([10, 10])
    fp = x0[10*10:].reshape([10, 321])
    for i in range(0,length):
        order="etainoshrd"
        actual = np.squeeze(pd.read_csv('data/train-words.txt', header = None).to_numpy())
        data = np.loadtxt('./data/train-img-%d.txt'%(i+1))
        log_feature_potential=feature_potential(data, fp)
        log_transition_potential=tp
        l=data.shape[0]
        forward_messages,backward_messages=message_passing(data, log_feature_potential, log_transition_potential)
        Z=0
        fix_var=0
        for j in range(0,10):
            Z=Z+np.exp(log_feature_potential[fix_var][j]+np.log(forward_messages[str(fix_var-1)+str(fix_var)+str(j)])+np.log(backward_messages[str(fix_var+1)+str(fix_var)+str(j)]))
        sumFP=0
        for j in range(0,data.shape[0]):
            sumFP=sumFP+log_feature_potential[j][order.index(actual[i][j])]
        sumTP=0
        for j in range(0,data.shape[0]-1):
            sumTP=sumTP+log_transition_potential[order.index(actual[i][j]), order.index(actual[i][j+1])]
        sums=sums+sumFP+sumTP-np.log(Z)
    sums=sums/length

    return sums

def get_log_test(x0):
    sums=0
    tp = x0[0:10*10].reshape([10, 10])
    fp = x0[10*10:].reshape([10, 321])
    for i in range(0,200):
        order="etainoshrd"
        actual = np.squeeze(pd.read_csv('data/test-words.txt', header = None).to_numpy())
        data = np.loadtxt('./data/test-img-%d.txt'%(i+1))
        log_feature_potential=feature_potential(data, fp)
        log_transition_potential=tp
        l=data.shape[0]
        forward_messages,backward_messages=message_passing(data, log_feature_potential, log_transition_potential)
        Z=0
        fix_var=0
        for j in range(0,10):
            Z=Z+np.exp(log_feature_potential[fix_var][j]+np.log(forward_messages[str(fix_var-1)+str(fix_var)+str(j)])+np.log(backward_messages[str(fix_var+1)+str(fix_var)+str(j)]))
        sumFP=0
        for j in range(0,data.shape[0]):
            sumFP=sumFP+log_feature_potential[j][order.index(actual[i][j])]
        sumTP=0
        for j in range(0,data.shape[0]-1):
            sumTP=sumTP+log_transition_potential[order.index(actual[i][j]), order.index(actual[i][j+1])]
        sums=sums+sumFP+sumTP-np.log(Z)
    sums=sums/200

    return sums

def objective(x0,length):
    log = question13(x0,length)
    
    return -log
def test_log(passing_val):
    
    return get_log_test(passing_val)
def feature_potential (x, params):
    #x is the array of length n by 321
    #feature params is 10 by 321
    log_potential=np.empty((x.shape[0],10))
    for i in range(0,x.shape[0]):
        for j in range(0,10):
            log_potential[i][j]= np.dot(x[i],params[j])
    return log_potential

def compute_pairwise_marginal(data,log_feature_potential,log_transition_potential,forward_messages,backward_messages,Z):
    pairwise_arr={}
    l=data.shape[0]
    for i in range(0,l-1):
        j=i+1
        pairwise=np.empty((10,10))
        for c1 in range(0,10):
            for c2 in range(0,10):
                pairwise[c1][c2]=np.exp(log_feature_potential[i][c1]+log_feature_potential[j][c2]+log_transition_potential[c1][c2]+np.log(forward_messages[str(i-1)+str(i)+str(c1)])+np.log(backward_messages[str(j+1)+str(j)+str(c2)]))
                pairwise[c1][c2]=pairwise[c1][c2]/Z
        pairwise_arr[str(i)+str(j)]=pairwise
    return pairwise_arr



def compute_single_marginal(data,log_feature_potential,log_transition_potential,forward_messages,backward_messages,Z):
    marginal=np.empty((data.shape[0],10))
    l=data.shape[0]
    for i in range(0,l):
        for c in range(0,10):
            marginal[i][c]=np.exp(log_feature_potential[i][c]+np.log(forward_messages[str(i-1)+str(i)+str(c)])
            +np.log(backward_messages[str(i+1)+str(i)+str(c)]))
            marginal[i][c]=marginal[i][c]/Z
    return marginal

def message_passing(data, log_feature_potential,log_transition_potential):
    # Message passing algorithm
    # input: (log-) potential
    # outputs: forward/backward messages

    l=data.shape[0]
    forward_messages={}
    for val in range(0,10):
        forward_messages["-10"+str(val)]=1
    for from_var in range(0,l-1):
        to_var=from_var+1
        for to_val in range(0,10):
            message_sum=0
            for from_val in range(0,10):
                message_sum = message_sum+np.exp(log_feature_potential[from_var][from_val]+log_transition_potential[from_val][to_val]
                +np.log(forward_messages[str(from_var-1)+str(from_var)+str(from_val)]))
            forward_messages[str(from_var)+str(to_var)+str(to_val)]=message_sum
    backward_messages={}
    for val in range(0,10):
        backward_messages[str(l)+str(l-1)+str(val)]=1
    for from_var in range(l-1,0,-1):
        to_var=from_var-1
        for to_val in range(0,10):
            message_sum=0
            for from_val in range(0,10):
                message_sum = message_sum+np.exp(log_feature_potential[from_var][from_val]+log_transition_potential[from_val][to_val]
                +np.log(backward_messages[str(from_var+1)+str(from_var)+str(from_val)]))  
                backward_messages[str(from_var)+str(to_var)+str(to_val)]=message_sum
    return forward_messages,backward_messages


def deriviate_obejctive(x0,length):
    tp = x0[0:10*10].reshape([10, 10])
    fp = x0[10*10:].reshape([10, 321])
    gradient_tp = np.zeros([10,10])
    gradient_fp = np.zeros([10,321])
    key2 = {'e': 0, 't': 1, 'a': 2, 'i': 3, 'n': 4, 'o': 5, 's': 6, 'h': 7, 'r': 8, 'd': 9}
    words = open('./data/train-words.txt', 'r').read().splitlines()
    order="etainoshrd"
    for i in range(length):
        data = np.loadtxt('./data/train-img-%d.txt'%(i+1))
       
        log_feature_potential=feature_potential(data, fp)
        Z=0
        fix_var=0
        
        forward_messages,backward_messages = message_passing(data,log_feature_potential,tp)

        for d in range(0,10):
            Z=Z+np.exp(log_feature_potential[fix_var][d]+np.log(forward_messages[str(fix_var-1)+str(fix_var)+str(d)])+np.log(backward_messages[str(fix_var+1)+str(fix_var)+str(d)]))
        
        marginal=compute_single_marginal(data,log_feature_potential,tp,forward_messages,backward_messages,Z)
        
        pairwise = compute_pairwise_marginal(data,log_feature_potential,tp,forward_messages,backward_messages,Z)

        #pair = pairwise(data,fp,tp,key2)
        for j in range(0, len(words[i])-1):
            l1 = words[i][j]
            l2 = words[i][j+1]
            gradient_tp[key2[l1]][key2[l2]] +=1
            gradient_tp -= pairwise[str(j)+str(j+1)].T
        
        for j in range(len(words[i])):
            l = words[i][j]
            gradient_fp[key2[l]] += data[j]
            gradient_fp -= marginal[j].reshape(10,1) * data[j]
       
    gradient_tp /= length
    gradient_fp /= length
    return -np.concatenate([gradient_tp.ravel(),gradient_fp.ravel()]) 
def accuracy(x0):
    pred=[]
    tp = x0[0:10*10].reshape([10, 10])
    fp = x0[10*10:].reshape([10, 321])
    for img in range(1,201):
        #img=i
        order="etainoshrd"
        data = np.loadtxt('./data/test-img-%d.txt'%(img))
        log_feature_potential=feature_potential(data, fp)
        log_transition_potential= tp
        l=data.shape[0]
        forward_messages, backward_messages=message_passing(data, log_feature_potential,log_transition_potential)
        Z=0
        fix_var=0
        for i in range(0,10):
            Z=Z+math.exp(log_feature_potential[fix_var][i]+math.log(forward_messages[str(fix_var-1)+str(fix_var)+str(i)])+math.log(backward_messages[str(fix_var+1)+str(fix_var)+str(i)]))
        marginal=compute_single_marginal(data,log_feature_potential,log_transition_potential,forward_messages,backward_messages,Z)
        pairwise_arr=compute_pairwise_marginal(data,log_feature_potential,log_transition_potential,forward_messages,backward_messages,Z)
        max_index_row = np.argmax(marginal, axis=1)
        predict="".join(list(map(lambda x: order[x],max_index_row)))
        pred.append(predict)
    pred=np.asarray(pred)
    actual = np.squeeze(pd.read_csv('data/test-words.txt', header = None).to_numpy())
    correct=0
    total=0
    for i in range(0,len(pred)):
        total=total+len(pred[i])
        if (pred[i]==actual[i]):
            correct=correct+len(pred[i])
        else:
            for j in range(0,len(pred[i])):
                if (pred[i][j]==actual[i][j]):
                    correct=correct+1
    return(correct/total)

if __name__ == '__main__':
    key1 = list("etainoshrd")
    key2 = {'e': 0, 't': 1, 'a': 2, 'i': 3, 'n': 4, 'o': 5, 's': 6, 'h': 7, 'r': 8, 'd': 9}
   
    #x_1= np.loadtxt('./data/test-img-1.txt')
    
    #x_2 = np.loadtxt('./data/test-img-2.txt')
    #x_3 = np.loadtxt('./data/test-img-3.txt')
    model_feature = np.loadtxt('./model/feature-params.txt')
    model_params = np.loadtxt('./model/transition-params.txt')
    
    train_size = [50, 100, 150, 200, 250, 300, 350, 400]
    timearr = []
    accuracyarr = []
    logtestarr =[] 
    
    for t in train_size:
        begin = time.time()
        x0 = np.zeros(3310)
        res = minimize(objective,x0, args=(t,), jac=deriviate_obejctive,method='L-BFGS-B')
        timearr.append(time.time()-begin)
        accuracyarr.append(1-accuracy(res.x))
        logtestarr.append(test_log(res.x))
    
        #accuracyarr.append()
    plt.plot(train_size,timearr)
    plt.xlabel('trainsize')
    plt.ylabel('train_time')
    plt.show()
    plt.plot(train_size,accuracyarr)
    plt.xlabel('trainsize')
    plt.ylabel('error rate')
    plt.show()
    plt.plot(train_size,logtestarr)
    plt.xlabel('trainsize')
    plt.ylabel('average conditional log-likelihood')
    plt.show()

    #[-2.799317567108158, -2.0090996109179287, -1.8913960670830905, -1.6957458763371596, -1.5667150936611016, -1.5357074354985027, -1.4066491574460291, -1.364439672394779]
   # print(q2(x_1,model_feature))

    