class Environment:
    def __init__(self, graph):
        self.graph = graph
        self.barrier = 0
        self.path = 1
        self.start = 2
        self.goal = 3
        self.carrot = 4
        self.lines = 9
        self.columns = 8
        self.num_actions = 4
        self.carrot_index = []
        self.barrier_index = []
        self.start_position = self.get_index(self.start, self.graph)[0]
        self.goal_position = self.get_index(self.goal, self.graph)[0]
        self.current = self.start_position
        self.pre_current = []
        self.graph_backup = copy.deepcopy(graph)

    def get_index(self, state, graph):
        index = []
        for i in range(self.lines):
            for j in range(self.columns):
                if state == graph[i][j]:
                    index.append([i, j])
        return index

    def is_carrot(self):
        self.carrot_index = self.get_index(self.carrot, self.graph)
    
    def is_barrier(self):
        self.barrier_index = self.get_index(self.barrier, self.graph)

    def check_line_path_r(self, current):
        carrot_cache = []
        barrier_cache = []
        goal = 0
        for i in range(current[1]+1, self.columns):
            if self.graph[current[0]][i] == self.carrot:
                carrot_cache.append(i)
            if self.graph[current[0]][i] == self.barrier:
                barrier_cache.append(i-1)
            if self.graph[current[0]][i] == self.goal:
                goal = self.goal_position[1]  
        if goal:
            return carrot_cache + barrier_cache + [goal]
        return carrot_cache + barrier_cache
    
    def check_line_path_l(self, current):
        carrot_cache = []
        barrier_cache = []
        goal = 0
        for i in range(0, current[1]):
            if self.graph[current[0]][i] == self.carrot:
                carrot_cache.append(i)
            if self.graph[current[0]][i] == self.barrier:
                barrier_cache.append(i+1)
            if self.graph[current[0]][i] == self.goal:
                goal = self.goal_position[1]  
        if goal==self.goal_position[1]:
            return carrot_cache + barrier_cache + [goal]
        return carrot_cache + barrier_cache
    
    def check_column_path_u(self, current):
        carrot_cache = []
        barrier_cache = []
        goal = 0
        for i in range(0, current[0]):
            if self.graph[i][current[1]] == self.carrot:
                carrot_cache.append(i)
            if self.graph[i][current[1]] == self.barrier:
                barrier_cache.append(i+1)
            if self.graph[i][current[1]] == self.goal:
                goal = self.goal_position[0]  
        if goal == self.goal_position[0]:
            return carrot_cache + barrier_cache + [goal]
        return carrot_cache + barrier_cache
    
    def check_column_path_d(self, current):
        carrot_cache = []
        barrier_cache = []
        goal = 0
        for i in range(current[0]+1, self.lines):
            if self.graph[i][current[1]] == self.carrot:
                carrot_cache.append(i)
            if self.graph[i][current[1]] == self.barrier:
                barrier_cache.append(i-1)
            if self.graph[i][current[1]] == self.goal:
                goal = self.goal_position[0]  
        if goal == self.goal_position[0]:
            return carrot_cache + barrier_cache + [goal]
        return carrot_cache + barrier_cache
    
    def check_neighbors(self, current):
        neighbors = []
        neighbors.append([current[0]-1, current[1]])
        neighbors.append([current[0]+1, current[1]])
        neighbors.append([current[0], current[1]-1])
        neighbors.append([current[0], current[1]+1])
        
        for i in range(len(neighbors)):
            # avoid out array 
            if i >= len(neighbors):
                break
            if (0 <= neighbors[i][0] <= self.lines-1 and 0 <= neighbors[i][1] <= self.columns-1):
                continue
            else:
                neighbors.remove(neighbors[i])
        return neighbors
    
    # check whether the current node can be moved
    def check_moved(self, current):
        current_neighbors = self.check_neighbors(current)
        cache = []
        for i in current_neighbors:
            if self.graph[i[0]][i[1]] == self.barrier:
                cache.append(self.barrier)
        if len(cache) == len(current_neighbors):
            return True
        return False
    
    def reset(self):
        self.state = self.start
        self.current = copy.deepcopy(self.start_position)
        return self.current
    
    # Manhattan distance to measure reward
    def distance(self, current, goal):
        reward = abs(goal[0]-current[0]) + abs(goal[1]-current[1])
        return -reward
    
    # calaulate the Q table 
    def step(self, action, pre_current):
        if action == 0:
            if self.current[1] == self.columns-1:
                self.current[1] = self.columns-1
            else:  
                line_state_r = self.check_line_path_r(self.current)
                if line_state_r:
                    self.current[1] = min(line_state_r)
                else:
                    self.current[1] = self.columns-1
            self.state = self.graph[self.current[0]][self.current[1]]
            for i in range(pre_current[1], self.current[1]):
                self.graph[self.current[0]][i] = self.barrier

        elif action == 1:
            if self.current[0] == 0:
                self.current[0] = 0
            else:  
                column_state_u = self.check_column_path_u(self.current)
                if column_state_u:
                    self.current[0] = max(column_state_u)
                else:
                    self.current[0] = 0
            self.state = self.graph[self.current[0]][self.current[1]]
            for i in range(self.current[0]+1, pre_current[0]+1):
                self.graph[i][self.current[1]] = self.barrier
            
        elif action == 2:
            if self.current[1] == 0:
                self.current[1] = 0
            else:  
                line_state_l = self.check_line_path_l(self.current)
                if line_state_l:
                    self.current[1] = max(line_state_l)
                else:
                    self.current[1] = 0
            self.state = self.graph[self.current[0]][self.current[1]]
            for i in range(self.current[1]+1, pre_current[1]+1):
                self.graph[self.current[0]][i] = self.barrier 

        elif action == 3:
            if self.current[0] == self.lines-1:
                self.current[0] = self.lines-1
            else:  
                column_state_d = self.check_column_path_d(self.current)
                if column_state_d:
                    self.current[0] = min(column_state_d)
                else:
                    self.current[0] = self.lines-1
            self.state = self.graph[self.current[0]][self.current[1]]
            for i in range(pre_current[0], self.current[0]):
                self.graph[i][self.current[1]] = self.barrier

        reward = self.distance(self.current, self.goal_position)
        done = self.state == self.goal

        return self.current, reward, done