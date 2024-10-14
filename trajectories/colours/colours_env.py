import gym
from gym import spaces
import numpy as np
from random import randint, shuffle
from PIL import Image
from collections import deque

class ColorsEnv(gym.Env):
    def __init__(self, env_name):
        super(ColorsEnv, self).__init__()
        self.SIZE = 5
        self.WORLD = np.zeros((self.SIZE, self.SIZE))

        self.COLORS = {
            1 : (0,   0,   0  ),
            2 : (255, 0,   0  ),
            3 : (0,   255, 0  ),
            4 : (0,   0,   255),
        }

        self.setup_world()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high = 5, shape=(self.SIZE, self.SIZE), dtype=int)

        self.state_dim = 11
        self.action_dim = 5
        self.env_type = 'colours'

        self.has_red = False
        self.has_green = False
        self.has_blue = False


    def setup_world(self):
        self.WORLD = np.zeros((self.SIZE, self.SIZE))
    
        coordinates = set()
        while len(coordinates) < 4:
            x = randint(*(0, self.SIZE - 1))
            y = randint(*(0, self.SIZE - 1))
            coordinates.add((x, y))
        coordinates = list(coordinates)

        for i, coord in enumerate(coordinates):
            self.WORLD[coord[0], coord[1]] = i + 1

    def reset(self):
        self.setup_world()
        self.has_red = False
        self.has_green = False
        self.has_blue = False
        return self.WORLD
        
    def get_char_pos(self, dir):
        #row col 
        directions = {
            0 : np.array([-1,  0]),  #UP
            1 : np.array([1 ,  0]),  #DOWN
            3 : np.array([0 ,  1]),  #RIGHT
            2 : np.array([0 , -1])   #LEFT
        }

        char_pos = np.array([np.where(self.WORLD == 1)[0][0], np.where(self.WORLD == 1)[1][0]])
        new_char_pos = char_pos + directions[dir]

        for i in range(2):
            if new_char_pos[i] < 0 or new_char_pos[i] >= self.SIZE:
                return char_pos
        return new_char_pos

    def move_char(self, dir):
        old_ = np.array([np.where(self.WORLD == 1)[0][0], np.where(self.WORLD == 1)[1][0]])
        new_ = self.get_char_pos(dir)
        
        self.WORLD[old_[0], old_[1]] = 0
        self.WORLD[new_[0], new_[1]] = 1

    def step(self, action):
        self.move_char(action)
        state = self.WORLD
        simple_state = get_simple_obs(state)
    
        done = (2 not in state) and (3 not in state) and (4 not in state)

        if (self.has_red == False) and (simple_state[4] == 1):
            self.has_red = True
            reward = 1 
        elif (self.has_green == False) and (simple_state[7] == 1):
            self.has_green = True
            reward = 1 
        elif (self.has_blue == False) and (simple_state[10] == 1):
            self.has_blue = True
            reward = 1 
        else:
            reward = -0.5
    

        return state, reward, done, {} 

    def render(self, mode='human'):        
        print(self.get_obs())

            
    def close(self):
        pass

def get_img_from_obs(obs):
    color_map = {
        2: (255, 0, 0),   #RED
        3: (0, 255, 0),   #GREEN
        4: (0, 0, 255),   #BLUE
        
        1: (0, 0, 0),   #CHAR
        0: (255, 255, 255),   #WHITE BG

    }
    image_data = np.zeros((obs.shape[0], obs.shape[1], 3), dtype=np.uint8)  
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            image_data[i, j] = color_map[obs[i, j]]
    image = Image.fromarray(image_data, 'RGB')
    # image_resized = image.resize((64, 64), Image.NEAREST)
    return image

def find_shortest_path(grid, goal_nb):
    # Define directions
    directions = {
        0: np.array([-1, 0]),  # UP
        1: np.array([1, 0]),   # DOWN
        2: np.array([0, -1]),  # LEFT
        3: np.array([0, 1])    # RIGHT
    }
    
    dir_map = {
        (0, 1): 3,  # RIGHT
        (1, 0): 1,  # DOWN
        (0, -1): 2, # LEFT
        (-1, 0): 0  # UP
    }
    
    rows, cols = len(grid), len(grid[0])
    
    start = None
    end = None
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                start = (r, c)
            elif grid[r][c] == goal_nb:
                end = (r, c)
    
    if not start or not end:
        return []

    queue = deque([(start[0], start[1], [])])  # (row, col, path)
    visited = set()
    visited.add(start)
    
    while queue:
        r, c, path = queue.popleft()
        
        if (r, c) == end:
            return path
        
        for dr, dc in directions.values():
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if grid[nr][nc] == 0 or grid[nr][nc] == goal_nb:  # Avoid obstacles
                    visited.add((nr, nc))
                    new_dir = dir_map[(dr, dc)]
                    queue.append((nr, nc, path + [new_dir]))
    
    return []

def get_coords(obs, search):
    pos = np.where(obs == search)   
    return [pos[0][0], pos[1][0]]

def find_colour_index(list, index):
    for i, l in enumerate(list):
        if l[index] == 1:
            return i

    

def add_in_pickup(obs_list:list, action_list:list):
    updated_obs = obs_list.copy()
    updated_acts = action_list.copy()
    colour_ind = {
        'red' : find_colour_index(obs_list, 4),
        'green' : find_colour_index(obs_list, 7),
        'blue' : find_colour_index(obs_list, 10)
    }

    colour_places = {
        'red' :   4,
        'green' : 7,
        'blue' :  10
    }

    #Sort it in ascending order 
    colour_ind = dict(sorted(colour_ind.items(), key=lambda item: item[1]))

    for i, col in enumerate(colour_ind):
        row_old = updated_obs[colour_ind[col] + i].copy()
        row_old[colour_places[col]] = 0
        updated_obs.insert((colour_ind[col] + i), row_old)

        updated_acts.insert(colour_ind[col] + i, 4)

    return updated_obs, updated_acts



def get_simple_obs(obs):
    # [agent_x, agent_y, dis_r_x, dis_r_y, has_red, dis_g_x,\\
    #  dis_g_y, has_green, dis_b_x, dis_r_x, has_blue ] 

    has_red = 2 in obs
    has_green = 3 in obs 
    has_blue = 4 in obs 

    agent = get_coords(obs, 1)
    red = get_coords(obs, 2) if has_red else [-1, -1]
    green = get_coords(obs, 3) if has_green else [-1, -1]
    blue = get_coords(obs, 4) if has_blue else [-1, -1]

    state = np.zeros(11)

    state[0] = agent[0] #agent x 
    state[1] = agent[1] #agent y 

    state[2] = red[0]  #distance red x 
    state[3] = red[1] #distance red y 
    state[4] = 0 if has_red else 1 #If have red in state

    state[5] = green[0]  #distance green x 
    state[6] = green[1]  #distance green y 
    state[7] = 0 if has_green else 1 #If have green in state

    state[8] = blue[0]  #distance blue x 
    state[9] = blue[1]  #distance blue y 
    state[10] = 0 if has_blue else 1 #If have blue in state

    return state


def run_episode(env, goals = [2, 3, 4]):
    obs = env.reset()
    shuffle(goals) #Randomise order of colours 
    done = False 
    ep_states = [get_simple_obs(obs.copy())]
    ep_actions = []
    ep_rewards = []
    ep_length = 0
    path_lengths = [0, 0, 0]
    path_i = 0

    for goal in goals:
        path = find_shortest_path(obs, goal)
        path_lengths[path_i] = len(path)

        for action in path: 
            obs, reward, done, _ = env.step(action)
            ep_states.append(get_simple_obs(obs.copy()))
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_length += 1

        path_i += 1

    ep_actions.append(-1)

    equi_paths = (len(set(path_lengths)) == 1) and (path_lengths[0] == 3)

    # ep_states, ep_actions = add_in_pickup(ep_states, ep_actions)

    ep_length = len(ep_states[:-1])

    return ep_states[:-1], ep_actions[:-1], ep_rewards, ep_length, done, equi_paths




def save_colours_demonstrations(nb_traces = 15000, max_steps = 12):
    env = ColorsEnv('colours')
    state_dim = 11

    data_states = np.zeros([nb_traces, max_steps, state_dim], dtype='float32')
    data_actions = np.zeros([nb_traces, max_steps], dtype='long')

    tn = 0 
    
    while tn < nb_traces:
        try: 
            states, actions, _, length, done, eq = run_episode(env)

            if (length == max_steps) and done :
                for i in range(length):
                    data_states[tn][i] = states[i]
                    data_actions[tn][i] = actions[i]
                tn += 1
        except:
            pass
    
    size = str(nb_traces).replace('0', '')
    
    np.save(f'trajectories/colours/{size}k_nopick_newabs_states', data_states)
    np.save(f'trajectories/colours/{size}k_nopick_newabs_actions', data_actions)


if __name__ == '__main__':
    # env = ColorsEnv('colours')
    # np.set_printoptions(formatter={'all':lambda x: f'{x:>5}'})
    
    # ep_states, ep_actions, ep_rewards, ep_length, done, equi = run_episode(env)

    # new_states, new_acts = add_in_pickup(ep_states, ep_actions)

    # for s in ep_states:
    #     print(s)
    
    # print()

    # for s,a  in zip(new_states[:-1], new_acts[:-1]):
    #     print(s, a)



    # for s, a in zip(ep_states, ep_actions):
    #     print(get_simple_obs(s), a)
    #     print()

    # print(done)

    save_colours_demonstrations(15000, 12)
   