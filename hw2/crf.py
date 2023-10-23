import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it

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
    model = {0:[1],1:[0,2],2:[1,3],3:[2]}
    log_message = [] 
    passing_val = {}
    for y in range(len(key2)):
       
        log_message.append(get_pair(x,y,fp,tp,m1,m2,model,len(key2),passing_val))
    print(log_message)
    return
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
if __name__ == '__main__':
    key1 = list("etainoshrd")
    key2 = {'e': 0, 't': 1, 'a': 2, 'i': 3, 'n': 4, 'o': 5, 's': 6, 'h': 7, 'r': 8, 'd': 9}
    x_1= np.loadtxt('./data/test-img-1.txt')
    
    x_2 = np.loadtxt('./data/test-img-2.txt')
    x_3 = np.loadtxt('./data/test-img-3.txt')
    model_feature = np.loadtxt('./model/feature-params.txt')
    model_params = np.loadtxt('./model/transition-params.txt')
    newarr = [x_1,x_2,x_3]
    print(q1(x_1,model_feature))
    print(q2(newarr,model_feature,model_params,key2))
    q3(newarr,model_feature,model_params,key2)
    q4(newarr,model_feature,model_params,key1)
    q5(x_1,model_feature,model_params,key2)
    q6(x_1,model_feature,model_params,key2,0,1)
    q6(x_1,model_feature,model_params,key2,1,0)
    q6(x_1,model_feature,model_params,key2,3,2)
    q6(x_1,model_feature,model_params,key2,2,1)
   
    q7(x_1,model_feature,model_params,key2)
    test_img_arr = []
    for i in range (1,201):
        test_img_arr.append(np.loadtxt('./data/test-img-%d.txt'%i))
    q8(test_img_arr,model_feature,model_params,key2,key1)
    train_img_arr =[]
    for i in range(1,51):
        train_img_arr.append(np.loadtxt('./data/train-img-%d.txt'%i))
    q9(train_img_arr,model_feature,model_params,key2)
    
   # 
   # print(q2(x_1,model_feature))

    