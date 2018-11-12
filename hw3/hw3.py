import numpy as np
import sys
import copy

car_num=0
obs_num=0
grid_size=0
obs_list=[]
car_start_list=[]
reward_v=[]
car_end_list=[]
dir_map=[[-1,0],[0,-1],[1,0],[0,1]]

def get_input(file_name):
    with open(file_name) as f:
        global grid_size,car_num,obs_num
        grid_size=int(f.readline().strip())
        car_num=int(f.readline().strip())
        obs_num=int(f.readline().strip())

        for _ in xrange(obs_num):
            x,y=f.readline().strip().split(',')
            obs_list.append((int(x),int(y)))

        for _ in xrange(car_num):
            x,y=f.readline().strip().split(',')
            car_start_list.append((int(x),int(y)))

        for _ in xrange(car_num):
            x,y=f.readline().strip().split(',')
            car_end_list.append((int(x),int(y)))

def get_init_array():
    new_q=[]
    new_v=[]
    for i in xrange(grid_size):
        new_q.append([])
        new_v.append([])
        for j in xrange(grid_size):
            new_q[i].append([]) 
            new_v[i].append([]) 
            for c in xrange(car_num):
                new_q[i][j].append([])
                new_v[i][j].append(0)
                for d in xrange(4):
                    new_q[i][j][c].append([])
    return new_q,new_v 


def init():
    global reward_v
    for i in xrange(grid_size):
        for j in xrange(grid_size):
            for c in xrange(car_num):
                for d in xrange(4):
                    
                    new_i = i+dir_map[d][0]
                    new_j = j+dir_map[d][1]
                    reward = -1 
                    if not is_valid(new_i,new_j):
                        new_i = i
                        new_j = j

                    if((new_i,new_j) == car_end_list[c]):
                        reward += 100
                    if((new_i,new_j) in obs_list):
                        reward -= 100

                    reward_v[i][j][c][d] = reward

def is_valid(x,y):
    return x>=0 and x<grid_size and y >=0 and y <grid_size

def get_new_v(v,cur_i,cur_j,cur_car,cur_dir,gama=0.9):
    if (cur_i,cur_j) == car_end_list[cur_car]:
        return v[cur_i][cur_j][cur_car]
    #reward=-1
    #if (cur_i,cur_j) in obs_list:
    #    reward-=100

    total_reward = 0
    for want_dir in xrange(4): 
        p = 0.1
        if want_dir == cur_dir:
            p = 0.7
        new_i = cur_i+dir_map[want_dir][0]
        new_j = cur_j+dir_map[want_dir][1]
        if not is_valid(new_i,new_j):
            new_i = cur_i
            new_j = cur_j
        #print("reward_v:",reward_v[cur_i][cur_j][cur_car][want_dir])
        total_reward += p*(reward_v[cur_i][cur_j][cur_car][want_dir]+gama*v[new_i][new_j][cur_car])
        '''
        if is_valid(new_i,new_j):
            total_reward+=p*(v[new_i][new_j][cur_car])
            print ("total :",cur_i,cur_j,want_dir, p, cur_dir, total_reward, v[new_i][new_j][cur_car])
        else:
            total_reward+=p*v[cur_i][cur_j][cur_car]
            #total_reward+=p*(reward+gama*v[cur_i][cur_j][cur_car])

            print ("total :",cur_i,cur_j,want_dir, p, cur_dir, total_reward, v[cur_i][cur_j][cur_car])
        '''
        #print ("total :",cur_i,cur_j,want_dir, p, cur_dir, total_reward, reward_v[cur_i][cur_j][cur_car][want_dir], v[new_i][new_j][cur_car])
    return total_reward
    #return gama*total_reward+reward_v[cur_i][cur_j][cur_car]

def run_step(q,v,new_q,new_v):

    for i in xrange(grid_size):
        for j in xrange(grid_size):
            for c in xrange(car_num):
                new_v[i][j][c]=None
                for d in xrange(4):
                    new_q[i][j][c][d] = get_new_v(v, i,j,c,d)
                    if(not new_v[i][j][c] or new_q[i][j][c][d]>new_v[i][j][c]):
                        new_v[i][j][c] = new_q[i][j][c][d]
    return new_q,new_v

def train():
    epoch = 0
    last_q,last_v=get_init_array()
    global reward_v
    reward_v=copy.copy(last_q)

    cur_q=last_q
    cur_v=last_v

    init()


    while epoch < 20:
        cur_q,cur_v=get_init_array()
        new_q,new_v = run_step(last_q,last_v,cur_q,cur_v)
        last_q=new_q
        last_v=new_v
        epoch +=1
        
        '''
        print ("epoch:",epoch)
        for i in xrange(grid_size):
            for j in xrange(grid_size):
                for c in xrange(car_num):
                    print ("i = %d j = %d c = %d" %(i,j,c),new_v[i][j][c])
                    for d in xrange(4):
                        print ("i = %d j = %d c = %d d= %d" %(i,j,c,d),new_q[i][j][c][d])

        '''
     

    return cur_q,cur_v

def get_policy(pos,cur_car,q):
    #return np.argmax(q[pos[0]][pos[1]][cur_car])

    max_score = None
    policy = None
    for d in [1,3,2,0]:
        if not max_score or q[pos[0]][pos[1]][cur_car][d] > max_score:
            #if( max_score and max_score == q[pos[0]][pos[1]][cur_car][d]))):
            #    print ("get policy:",pos, cur_car, "d:",d, "val:",q[pos[0]][pos[1]][cur_car][d], "max score:",max_score)
            #    continue
            max_score = q[pos[0]][pos[1]][cur_car][d]
            policy = d
    return policy

def turn_left(move):
    return (move+1)%4

def turn_right(move):
    return (move+3)%4

def go(pos, move, cur_car):
    cur_i,cur_j = pos

    new_i = cur_i+dir_map[move][0]
    new_j = cur_j+dir_map[move][1]
    reward = 0
    if not is_valid(new_i,new_j):
        new_i = cur_i
        new_j = cur_j
    #print ("pos:",pos,"go:",(new_i,new_j), "reward:",reward_v[cur_i][cur_j][cur_car][move])
    return (new_i,new_j), reward_v[cur_i][cur_j][cur_car][move]+reward

def get_score(q):

    for i in range(car_num):
        total_reward = []
        for j in range(10):
            pos = car_start_list[i]
            np.random.seed(j+1)
            sw = np.random.random_sample(1000000)
            k=0
            score = 0.0
            #print ("counrd:",j)
            while pos !=car_end_list[i]:
                move = get_policy(pos,i,q)
                if sw[k]>0.7:
                    if sw[k]>0.8:
                        if sw[k] >0.9:
                            move = turn_left(turn_left(move))
                        else:
                            move = turn_left(move)
                    else:
                        move = turn_right(move)
                k+=1
                pos, reward = go(pos,move,i)
                score += reward
            #print ("get score:",score)
            total_reward.append(score)
        print ("get avg:",np.mean(total_reward))

if __name__ == '__main__':
    file_name=sys.argv[1]
    get_input(file_name)
    
    print ("car num:",car_num,"ob num:",obs_num,"grid_size:",grid_size)
    print ("ob list:",obs_list, "car start:",car_start_list,"car end:",car_end_list)

    q,v = train()

    '''
    print ("reward:")
    for c in xrange(car_num):
        for i in xrange(grid_size):
            for j in xrange(grid_size):
                for d in xrange(4):
                    print ("i = %d j = %d c = %d d= %d" %(i,j,c,d),reward_v[i][j][c][d])
    '''

    for c in xrange(car_num):
        for i in xrange(grid_size):
            for j in xrange(grid_size):
                print ("i = %d j = %d c = %d" %(i,j,c),v[i][j][c])
                for d in xrange(4):
                    print ("i = %d j = %d c = %d d= %d" %(i,j,c,d),q[i][j][c][d])
                print ("i = %d j = %d dir = %d" %(i,j,np.argmax(q[i][j][c])))

    for i in xrange(car_num):
        print ("score:",v[car_start_list[i][0]][car_start_list[i][1]][i])
        
    get_score(q)             
