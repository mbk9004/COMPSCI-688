import math 
import itertools
import statistics



vals = {'A' : (1,2,3), 'G' : (1,2), 'C' : (1,2,3,4), 'B' : (1,2),'H' : (1,2), 'E' : (1,2),'R' : (1,2), 'I' : (1,2), 'D' : (1,2) }
order = ['A', 'G', 'C', 'B', 'H', 'E', 'R','I', 'D']

parents_node = {}

counts ={}
parents_node['A'] = []
parents_node['G'] = []
parents_node['C'] = ['D']
parents_node['B'] = ['A', 'G']
parents_node['H'] = ['A', 'G']
parents_node['E'] = ['D']
parents_node['R'] = ['A', 'D']
parents_node['I'] =['D']
parents_node['D'] = ['B','H']



count_bay ={'A': {1: 0, 2: 0, 3: 0}, 
            'G': {1: 0, 2: 0}, 
            'C': {(1,): 0, (1, 1): 0, (2, 1): 0, (3, 1): 0, (4, 1): 0, (2,): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0}, 
            'B': {(1, 1): 0, (1, 1, 1): 0, (2, 1, 1): 0, (1, 2): 0, (1, 1, 2): 0, (2, 1, 2): 0, (2, 1): 0, (1, 2, 1): 0, (2, 2, 1): 0, (2, 2): 0, (1, 2, 2): 0, (2, 2, 2): 0, (3, 1): 0, (1, 3, 1): 0, (2, 3, 1): 0, (3, 2): 0, (1, 3, 2): 0, (2, 3, 2): 0},
              'H': {(1, 1): 0, (1, 1, 1): 0, (2, 1, 1): 0, (1, 2): 0, (1, 1, 2): 0, (2, 1, 2): 0, (2, 1): 0, (1, 2, 1): 0, (2, 2, 1): 0, (2, 2): 0, (1, 2, 2): 0, (2, 2, 2): 0, (3, 1): 0, (1, 3, 1): 0, (2, 3, 1): 0, (3, 2): 0, (1, 3, 2): 0, (2, 3, 2): 0}, 
              'E': {(1,): 0, (1, 1): 0, (2, 1): 0, (2,): 0, (1, 2): 0, (2, 2): 0},
                'R': {(1, 1): 0, (1, 1, 1): 0, (2, 1, 1): 0, (1, 2): 0, (1, 1, 2): 0, (2, 1, 2): 0, (2, 1): 0, (1, 2, 1): 0, (2, 2, 1): 0, (2, 2): 0, (1, 2, 2): 0, (2, 2, 2): 0, (3, 1): 0, (1, 3, 1): 0, (2, 3, 1): 0, (3, 2): 0, (1, 3, 2): 0, (2, 3, 2): 0}, 
                'I': {(1,): 0, (1, 1): 0, (2, 1): 0, (2,): 0, (1, 2): 0, (2, 2): 0}, 
                'D': {(1, 1): 0, (1, 1, 1): 0, (2, 1, 1): 0, (1, 2): 0, (1, 1, 2): 0, (2, 1, 2): 0, (2, 1): 0, (1, 2, 1): 0, (2, 2, 1): 0, (2, 2): 0, (1, 2, 2): 0, (2, 2, 2): 0}}


	
def init():
    
    count_bay = {'A': {1: 0, 2: 0, 3: 0}, 
            'G': {1: 0, 2: 0}, 
            'C': {(1,): 0, (1, 1): 0, (2, 1): 0, (3, 1): 0, (4, 1): 0, (2,): 0, (1, 2): 0, (2, 2): 0, (3, 2): 0, (4, 2): 0}, 
            'B': {(1, 1): 0, (1, 1, 1): 0, (2, 1, 1): 0, (1, 2): 0, (1, 1, 2): 0, (2, 1, 2): 0, (2, 1): 0, (1, 2, 1): 0, (2, 2, 1): 0, (2, 2): 0, (1, 2, 2): 0, (2, 2, 2): 0, (3, 1): 0, (1, 3, 1): 0, (2, 3, 1): 0, (3, 2): 0, (1, 3, 2): 0, (2, 3, 2): 0},
              'H': {(1, 1): 0, (1, 1, 1): 0, (2, 1, 1): 0, (1, 2): 0, (1, 1, 2): 0, (2, 1, 2): 0, (2, 1): 0, (1, 2, 1): 0, (2, 2, 1): 0, (2, 2): 0, (1, 2, 2): 0, (2, 2, 2): 0, (3, 1): 0, (1, 3, 1): 0, (2, 3, 1): 0, (3, 2): 0, (1, 3, 2): 0, (2, 3, 2): 0}, 
              'E': {(1,): 0, (1, 1): 0, (2, 1): 0, (2,): 0, (1, 2): 0, (2, 2): 0},
                'R': {(1, 1): 0, (1, 1, 1): 0, (2, 1, 1): 0, (1, 2): 0, (1, 1, 2): 0, (2, 1, 2): 0, (2, 1): 0, (1, 2, 1): 0, (2, 2, 1): 0, (2, 2): 0, (1, 2, 2): 0, (2, 2, 2): 0, (3, 1): 0, (1, 3, 1): 0, (2, 3, 1): 0, (3, 2): 0, (1, 3, 2): 0, (2, 3, 2): 0}, 
                'I': {(1,): 0, (1, 1): 0, (2, 1): 0, (2,): 0, (1, 2): 0, (2, 2): 0}, 
                'D': {(1, 1): 0, (1, 1, 1): 0, (2, 1, 1): 0, (1, 2): 0, (1, 1, 2): 0, (2, 1, 2): 0, (2, 1): 0, (1, 2, 1): 0, (2, 2, 1): 0, (2, 2): 0, (1, 2, 2): 0, (2, 2, 2): 0}}
    return count_bay

def count_number(count,file):
    count_dict = count
    
    for line in open(file,'r'):
        temp_dict ={}
        temp = line.split(',')
        temp[-1] = temp[-1].strip()
        temp = list(map(int, temp))
        for x in range(len(temp)):
            temp_dict[order[x]] = temp[x]
        #print(temp_dict)
        for key, value in parents_node.items():
            if len(value)!=0:
                tlist = []
                for v in value:
                    tlist.append(temp_dict[v])
                t = tuple(tlist)
                
                n = (temp_dict[key], ) + t
                count_dict[key][n] +=1
                count_dict[key][t] +=1 
            
            else:
                count_dict[key][temp_dict[key]] +=1
            
    return count_dict         

file1 = 'data/data-train-1.txt'
theta_test = init()
def q5(c,filename):
    total= 0
    count = count_number(c,filename)
    
    with open(filename, 'r') as fp:
        total = len(fp.readlines())
        fp.close()
    
    avg = 0
    for o in order:
        prior = parents_node[o]
        if len(prior) == 0:
            n = 0
            for v in vals[o]:
               theta = count[o][v]/total 
               theta_test[o][v] = theta
               avg += theta * math.log(theta)
        else:
           
            pa = []
            for p in parents_node[o]:
                pa.append(vals[p])
            for p in itertools.product(*pa):
                p_total = count[o][p]
                #print(p_total)
                if p_total == 0 : 
                    continue
                for v in vals[o]:
                    n = (v,)+p
                    number= count[o][n]
                   
                    
                    theta = number/p_total
                    theta_test[o][n] = theta
                    
                    front_mult = float(number/total)
                    avg += front_mult * math.log(theta)
    return avg

def test_avg(c,filename):
    total= 0
    count = count_number(c,filename)
   
    with open(filename, 'r') as fp:
        total = len(fp.readlines())
        fp.close()
    avg = 0
    for o in order:
        prior = parents_node[o]
        if len(prior) == 0:
            n = 0
            for v in vals[o]:
               front = count[o][v]/total 
               theta = theta_test[o][v]
               avg += front * math.log(theta)
        else:

            pa = []
            for p in parents_node[o]:
                pa.append(vals[p])
            for p in itertools.product(*pa):
                p_total = count[o][p]
                #print(p_total)
                if p_total == 0 : 
                    continue
                for v in vals[o]:
                    n = (v,)+p
                    number= count[o][n]
                    theta = theta_test[o][n]
                    front_mult = float(number/total)
                    avg += front_mult * math.log(theta)
    return avg


def joint_p(dictionary,counter):
    
    prob_data = []
    prob =1 
    for key, value in parents_node.items():
       
        if len(value) !=0:
            tlist = []
            for v in value:
                tlist.append(dictionary[v])
            t = tuple(tlist)
            n = (dictionary[key], ) + t 
            prob_data.append(counter[key][n]/counter[key][t])
        else:
            n = 0
            for v in vals[key]:
                n += counter[key][v]
            prob_data.append(counter[key][dictionary[key]]/n)
    for i in prob_data:
        prob *= i

    return prob

def q7():
    count_bay = init()
    c1 = count_bay
   
    counter = count_number(c1,'data/data-train-1.txt')
    '''
    P (E = 1|A = 2, G = 1, C = 1, B = 1, H = 2, R = 2, I = 1, D = 1)
    P (B = 1|A = 3, G = 1, C = 3, H = 1, E = 2, R = 1, I = 1, D = 2)
    P (A = 1, H = 1|G = 2, C = 3, B = 2, E = 1, R = 2, I = 1, D = 1)
    P (D = 2|A = 1, G = 2, C = 1, B = 1, H = 1, E = 2, I = 2)
    '''
    q1_dict1 = {'A' : 2, 'G' : 1, 'C' : 1, 'B' : 1,'H' : 2, 'E' : 1,'R' : 2, 'I' : 1, 'D' : 1 }
    q1_dict2 = {'A' : 2, 'G' : 1, 'C' : 1, 'B' : 1,'H' : 2, 'E' : 2,'R' : 2, 'I' : 1, 'D' : 1 }
    x1 =joint_p(q1_dict1,counter)
    y1 = x1+joint_p(q1_dict2,counter)
    q1 =x1/y1
    
    q2_dict1 = {'A' : 3, 'G' : 1, 'C' : 3, 'B' : 1,'H' : 1, 'E' : 2,'R' : 1, 'I' : 1, 'D' : 2}
    q2_dict2 = {'A' : 3, 'G' : 1, 'C' : 3, 'B' : 2,'H' : 1, 'E' : 2,'R' : 1, 'I' : 1, 'D' : 2}
    x2 = joint_p(q2_dict1,counter)
    y2 = x2+joint_p(q2_dict2,counter)
    q2 = x2/y2
    
    q3_dict1 = {'A' : 1, 'G' : 2, 'C' : 3, 'B' : 2,'H' : 1, 'E' : 1,'R' : 2, 'I' : 1, 'D' : 1}
    q3_dict2 = {'A' : 1, 'G' : 2, 'C' : 3, 'B' : 2,'H' : 2, 'E' : 1,'R' : 2, 'I' : 1, 'D' : 1}
    q3_dict3 = {'A' : 2, 'G' : 2, 'C' : 3, 'B' : 2,'H' : 1, 'E' : 1,'R' : 2, 'I' : 1, 'D' : 1}
    q3_dict4 = {'A' : 2, 'G' : 2, 'C' : 3, 'B' : 2,'H' : 2, 'E' : 1,'R' : 2, 'I' : 1, 'D' : 1}
    q3_dict5 = {'A' : 3, 'G' : 2, 'C' : 3, 'B' : 2,'H' : 1, 'E' : 1,'R' : 2, 'I' : 1, 'D' : 1}
    q3_dict6 = {'A' : 3, 'G' : 2, 'C' : 3, 'B' : 2,'H' : 2, 'E' : 1,'R' : 2, 'I' : 1, 'D' : 1}
    x3 = joint_p(q3_dict1,counter)
    y3 = x3 + joint_p(q3_dict2,counter)+joint_p(q3_dict3,counter) +joint_p(q3_dict4,counter) + joint_p(q3_dict5,counter) + joint_p(q3_dict6,counter)
    q3 = x3/y3

    q4_dict1 =  {'A' : 1, 'G' : 2, 'C' : 1, 'B' : 1,'H' : 1, 'E' : 2,'R' : 1, 'I' : 2, 'D' : 2}
    q4_dict2 =  {'A' : 1, 'G' : 2, 'C' : 1, 'B' : 1,'H' : 1, 'E' : 2,'R' : 2, 'I' : 2, 'D' : 2}
    q4_dict3 =  {'A' : 1, 'G' : 2, 'C' : 1, 'B' : 1,'H' : 1, 'E' : 2,'R' : 1, 'I' : 2, 'D' : 1}
    q4_dict4 =  {'A' : 1, 'G' : 2, 'C' : 1, 'B' : 1,'H' : 1, 'E' : 2,'R' : 2, 'I' : 2, 'D' : 1}
    x4 = joint_p(q4_dict1,counter) + joint_p(q4_dict2,counter)
    y4 = x4+ joint_p(q4_dict3,counter) + joint_p(q4_dict4,counter)
    q4 = x4/y4
    return q1,q2,q3,q4




def q8():
   
    train_filename1 = 'data/data-train-1.txt'
    train_filename2 = 'data/data-train-2.txt'
    train_filename3 = 'data/data-train-3.txt'
    train_filename4 = 'data/data-train-4.txt'
    train_filename5 = 'data/data-train-5.txt'
    test_filename1 = 'data/data-test-1.txt'
    test_filename2 = 'data/data-test-2.txt'
    test_filename3 = 'data/data-test-3.txt'
    test_filename4 = 'data/data-test-4.txt'
    test_filename5 = 'data/data-test-5.txt'
    train_set = [train_filename1,train_filename2,train_filename3,train_filename4,train_filename5]
    test_set = [test_filename1,test_filename2,test_filename3,test_filename4,test_filename5]
    result_train = []
    result_test= []
    c = init()
    for t in range(len(train_set)):
        c = init()
        theta_test = init()
        result_train.append(q5(c,train_set[t]))
        
        d = init()
        result_test.append(test_avg(d,test_set[t]))

       
        
    train_mean =0
    test_mean =0
    print("train                  test")
    for i in range(len(result_train)):
        print(result_train[i],result_test[i])
        train_mean += result_train[i]
        test_mean += result_test[i]
    
    print("mean and std for training")
    print(train_mean/len(result_train))
    print(statistics.stdev(result_train))
    print("mean and std for testing")
    print(test_mean/len(result_test))
    print(statistics.stdev(result_test))


def classsificaiton_q9():
    test_filename1 = 'data/data-test-1.txt'
    test_filename2 = 'data/data-test-2.txt'
    test_filename3 = 'data/data-test-3.txt'
    test_filename4 = 'data/data-test-4.txt'
    test_filename5 = 'data/data-test-5.txt'
    #(d|a(n), g(n), c(n), b(n), h(n), e(n), r(n), i(n)).
    train_filename1 = 'data/data-train-1.txt'
    train_filename2 = 'data/data-train-2.txt'
    train_filename3 = 'data/data-train-3.txt'
    train_filename4 = 'data/data-train-4.txt'
    train_filename5 = 'data/data-train-5.txt'
    train_set = [train_filename1,train_filename2,train_filename3,train_filename4,train_filename5]
    test_set = [test_filename1,test_filename2,test_filename3,test_filename4,test_filename5]
    counters = []
    result = []
    for t in train_set:
        count = init()
        counters.append(count_number(count,t))
    for t in range(len(test_set)):
        correct = 0
        total = 0 
        for line in open(test_set[t],'r'):
            total +=1
            temp_dict ={}
            temp = line.split(',')
            temp[-1] = temp[-1].strip()
            temp = list(map(int, temp))
            for x in range(len(temp)):
                temp_dict[order[x]] = temp[x]
            
            original = temp_dict
            comparecase = 0
            temp_dict['D'] = 1 
            case1 = joint_p(temp_dict,counters[t])
            temp_dict['D'] = 2 
            case2 = joint_p(temp_dict,counters[t])
            if case1>case2:
                comparecase =1
            else:
                comparecase =2
           
            if comparecase == temp[-1]:
                correct +=1
        result.append(correct/total)
    
    return result,statistics.mean(result),statistics.stdev(result)
        

if __name__ == '__main__':
    cc = count_bay
    print(q5(cc,file1))
    print(q7())
    q8()
    print(classsificaiton_q9())