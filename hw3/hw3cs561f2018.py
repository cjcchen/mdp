import numpy as np
import sys
import copy
import time

car_num=0
obs_num=0
grid_size=0
obs_list=[]
car_start_list=[]
reward_v=[]
car_end_list=[]
dir_map=[[-1,0],[0,-1],[1,0],[0,1]]
esp =0.1
possible_move=[]
move_probobility=np.zeros((4,4),dtype=np.float)

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

def init():
    global reward_v,possible_move,move_probobility
    reward_v = np.ones((car_num, grid_size,grid_size))
    reward_v = reward_v * -1

    for c in xrange(car_num):
        reward_v[c][car_end_list[c][0]][car_end_list[c][1]] += 100

        for pos in obs_list:
            reward_v[c][pos[0]][pos[1]] -= 100

    possible_move=[]
    for i in xrange(grid_size):
        possible_move.append([])
        for j in xrange(grid_size):
            possible_move[i].append([])
            possible_move[i][j] = []
            for want_dir in xrange(4): 
                next_i,next_j = move_next((i,j),want_dir)
                possible_move[i][j].append((next_i,next_j))

    move_probobility+=0.1
    for i in xrange(4):
        move_probobility[i][i]=0.7

def move_next(pos,move_dir):
    def is_valid(x,y):
        return x>=0 and x<grid_size and y >=0 and y <grid_size
    x,y = pos
    next_x = x+dir_map[move_dir][0]
    next_y = y+dir_map[move_dir][1]
    if not is_valid(next_x,next_y):
        next_x = x
        next_y = y
    return next_x,next_y

def get_neighboor(v,cur_i,cur_j,cur_car):
    if (cur_i,cur_j) == car_end_list[cur_car]:
        return np.array([0]*4)
    move_list=[]
    for want_dir in xrange(4): 
        new_i,new_j = possible_move[cur_i][cur_j][want_dir]
        move_list.append(v[new_i][new_j])
    return np.array(move_list)

def run_step(v,car_idx,gama):
    neighbours = np.zeros((grid_size,grid_size,4),dtype=np.double)
    for i in xrange(grid_size):
        for j in xrange(grid_size):
            neighbours[i][j] = get_neighboor(v,i,j,car_idx)
    new_q = np.dot(neighbours, move_probobility)
    new_v = gama*np.max(new_q, axis=2) + reward_v[car_idx]
    return new_v,new_q

def train_one(car_idx,gama=0.9):
    def is_stop(last_v, cur_v):
        max_dis = np.max(np.abs(last_v-cur_v))
        return max_dis < esp*(1-gama)/gama

    last_v = np.zeros((grid_size,grid_size),dtype=np.float)
    while True:
        cur_v,cur_q=run_step(last_v,car_idx,gama)
        if is_stop(cur_v,last_v):
            break
        last_v = cur_v
    return cur_v,cur_q

def train():
    init()
    cur_q=[]
    cur_v=[]
    for i in xrange(car_num):
        v,q=train_one(i)
        cur_v.append(v)
        cur_q.append(q)
    return cur_q,cur_v

def simulate(q,v):
    def get_policy(pos,cur_car,q,v):
        #pre=[0,1,2,3]
        pre=[1,3,0,2]
        pp=np.argmax( [q[cur_car][pos[0]][pos[1]][d] for d in pre])
        return pre[pp]

    def turn_left(move):
        return (move+1)%4

    def turn_right(move):
        return (move+3)%4

    def go(pos, move, cur_car):
        next_i,next_j = move_next(pos,move)
        return (next_i,next_j), reward_v[cur_car][next_i][next_j]

    avg_score=[]
    for i in range(car_num):
        total_reward = []
        for j in range(10):
            pos = car_start_list[i]
            np.random.seed(j)
            sw = np.random.random_sample(1000000)
            k=0
            score = 0.0
            while pos !=car_end_list[i]:
                move = get_policy(pos,i,q,v)
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
    file_name="input.txt"
    get_input(file_name)
    
    q,v = train()
    avg_scores=simulate(q,v)             
    with open('output.txt','w') as f:
        for score in avg_scores: 
            f.write("%d\n" %np.floor(score))
    f.close()

