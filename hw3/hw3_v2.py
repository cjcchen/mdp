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
esp =0.1

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
                reward = -1 
                if((i,j) == car_end_list[c]):
                    reward += 100
                if((i,j) in obs_list):
                    reward -= 100

                reward_v[i][j][c] = reward
def is_stop(last_v, cur_v):
    max_dis = 0.0
    for i in xrange(grid_size):
        for j in xrange(grid_size):
            for c in xrange(car_num):
                d = last_v[i][j][c] - cur_v[i][j][c]
                if d > 0:
                    max_dis = max(d,max_dis)
                else:
                    max_dis = max(-d,max_dis)
    return max_dis < esp

def is_valid(x,y):
    return x>=0 and x<grid_size and y >=0 and y <grid_size

def get_new_v(v,cur_i,cur_j,cur_car,cur_dir,gama=0.9):
    if (cur_i,cur_j) == car_end_list[cur_car]:
        return 0

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
        total_reward += p*v[new_i][new_j][cur_car]
    return total_reward

def run_step(q,v,new_q,new_v, gama=0.9):

    for c in xrange(car_num):
        for i in xrange(grid_size):
            for j in xrange(grid_size):
                for d in xrange(4):
                    new_q[i][j][c][d] = get_new_v(v, i,j,c,d, gama)
                    if(not new_v[i][j][c] or new_q[i][j][c][d]>new_v[i][j][c]):
                        new_v[i][j][c] = new_q[i][j][c][d]
                new_v[i][j][c] = gama*new_v[i][j][c] + reward_v[i][j][c]
                print ("car %d i %d j %d:"%(c,i,j),new_q[i][j][c])
    return new_q,new_v

def train():
    epoch = 0
    last_q,last_v=get_init_array()
    global reward_v
    reward_v=copy.copy(last_q)
    init()
    cur_q,cur_v=get_init_array()

    #cur_v = copy.copy(reward_v)

    #while is_stop(last_v,cur_v):
    while epoch < 20:
        last_q=cur_q
        last_v=cur_v
        cur_q,cur_v=get_init_array()
        new_q,new_v = run_step(last_q,last_v,cur_q,cur_v)
        last_q=new_q
        last_v=new_v
        epoch +=1
        
        
    for c in xrange(car_num):
        for i in xrange(grid_size):
            for j in xrange(grid_size):
                print ("i = %d j = %d c = %d" %(i,j,c),cur_v[i][j][c])
                for d in xrange(4):
                    print ("i = %d j = %d c = %d d= %d" %(i,j,c,d),cur_q[i][j][c][d])
                print ("i = %d j = %d dir = %d" %(i,j,np.argmax(cur_q[i][j][c])))


    return cur_q,cur_v

def get_policy(pos,cur_car,q,v):
    max_score = None
    policy = None
    for d in [1,3,2,0]:
        if(pos == car_end_list[cur_car]):
            score = reward_v[pos[0]][pos[1]][cur_car]
        else:
            score = q[pos[0]][pos[1]][cur_car][d] 
        if not max_score or score > max_score:
            max_score = score
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
    return (new_i,new_j), reward_v[new_i][new_j][cur_car]

def simulate(q,v):

    avg_score=[]
    for i in range(car_num):
        total_reward = []
        for j in range(10):
            pos = car_start_list[i]
            np.random.seed(j)
            sw = np.random.random_sample(1000000)
            k=0
            score = 0.0
            print ("round:",j)
            while pos !=car_end_list[i]:
                move = get_policy(pos,i,q,v)
                print ("move:",move)
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
            total_reward.append(score)
        avg_score.append(np.mean(total_reward))
    return avg_score

if __name__ == '__main__':
    file_name=sys.argv[1]
    get_input(file_name)
    
    q,v = train()
    avg_scores=simulate(q,v)             
    for score in avg_scores: 
        print int(np.floor(score))
