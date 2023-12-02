# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import heapq
import logging


max_depth = 3

@register_agent("student_agent")
class StudentAgent(Agent):

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.aggression_heuristic = 2


    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()

        r, c = my_pos
        allowed_moves = {}
        print(f"starting s is {(0,r,c,-1)}")
        self.get_successors(allowed_moves, 0,chess_board,my_pos, adv_pos, max_step, -1)
        #self.get_successors((0,r,c,-1), allowed_moves, 0, chess_board, adv_pos, max_step, visited_tiles)
        print(allowed_moves)
        allowed_pos_list = list(allowed_moves)
        print(allowed_pos_list[0])
        next_r, next_c = allowed_pos_list[0]
        next_dir = allowed_moves[allowed_pos_list[0]][0]
        #heapq.heapify(allowed_moves_list)
        #next_r,next_c, next_dir = heapq.heappop(allowed_moves)
        #print(next_r,next_c, next_dir)

        #next_move = self.max_value((0,r,c,-1), chess_board, float('-inf'), float('inf'), adv_pos, max_step, 0)
        #print(f"next move is {next_move}")
        #heuristic, row, column, dir = next_move
        #next_position = row,column

        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return (next_r, next_c), next_dir
        #next_position, dir
    
    #my_pos, self.dir_map["u"]

    #self.dir_map["u"]
    def get_successors(self, allowed_moves, i,chess_board,my_pos, adv_pos, max_step, prev_dir):
        #returns a dictionary that contains available moves in the following format:
        #valid_position:tuple : valid_dirs_to_place_walls: list

        #make sure we don't consider moving back
        if prev_dir!=-1:
            prev_dir = self.opposites[prev_dir]

        #stop making moves when reached the max_step
        if i > max_step:
            return

        r, c = my_pos
        to_skip_pos = False

        # Build a list of the moves we can make
        allowed_dirs = [ d                                
            for d in range(0,4)                           # 4 moves possible
            if not chess_board[r,c,d] and                 # chess_board True means wall
            not adv_pos == (r+self.moves[d][0],c+self.moves[d][1])] # cannot move through Adversary

        if len(allowed_dirs)==0:
            # If no possible move, we must be enclosed by our Adversary
            return 

        if len(allowed_dirs)==1 and (i==0 or len(allowed_moves)!=0):
            # don't add the move that puts you in this position to possible moves dict
            to_skip_pos = True

        for dir in allowed_dirs:
            print(f"move is {(r,c)} and dir is {dir}")
            print(f"allowed_dirs are {allowed_dirs}")

            if not to_skip_pos:
                # adding the current tile and the possible directions for the wall on the dict of possible moves
                if (r,c) in allowed_moves:
                    print("move is already in dict")
                    if len(allowed_moves[(r,c)]) != len(allowed_dirs):
                        allowed_moves[(r,c)].append(dir)
                else:
                    print("move is not in dict")
                    allowed_moves[(r,c)] = [dir]

            #actually making the move and recursion for the remaining steps
            #when making the next move, make sure you're not moving to the direction agent was coming from 
            if dir!=prev_dir:
                m_r, m_c = self.moves[dir]
                next_pos = (r + m_r, c + m_c)
                self.get_successors(allowed_moves, i+1, chess_board,next_pos, adv_pos, max_step, dir)

    def get_successors2(self, s, allowed_moves, i,chess_board, adv_pos, max_step, visited_tiles):

        #i is the current step

        heuristic, r, c, prev_dir = s
        #visited_tiles.add((r,c))
        
        #make sure we don't consider moving back
        if prev_dir!=-1:
            prev_dir = self.opposites[prev_dir]

        #stop making moves when reached the max_step
        if i > max_step:
            return 0

        # Build a list of the moves we can make
        allowed_dirs = [ d                                
            for d in range(0,4)                           # 4 moves possible
            if not chess_board[r,c,d] and                 # chess_board True means wall
            not adv_pos == (r+self.moves[d][0],c+self.moves[d][1]) and
            d!=prev_dir] # cannot move through Adversary
        
        
        if len(allowed_dirs)==0:
            # If no possible move after making a move, we must be enclosed by our Adversary don't make that move
            return 
        
        if len(allowed_dirs)==1 and len(allowed_moves)!=0:
            #maybe add a check if the heap is empty
            return 
        
        heuristic = np.random.randint(0, len(chess_board))
        
        for dir in allowed_dirs:
            heapq.heappush(allowed_moves, (-heuristic, r,c,dir))
            #allowed_moves.append((r,c,dir))
            m_r, m_c = self.moves[dir]
            next_s = (heuristic, r + m_r, c + m_c, dir)

            #if((r + m_r, c + m_c) not in visited_tiles):
            self.get_successors(next_s, allowed_moves, i+1, chess_board, adv_pos, max_step, visited_tiles)
            

    def alpha_beta(self, chess_board, my_pos, adv_pos, max_depth):



        return
        


    def max_value(self, s, chess_board, alpha, beta, adv_pos, max_step, max_turn_count):
        print("in max_value")
        print(f"s is {s}")

        if max_turn_count==max_depth:
            return -s[0],s[1],s[2], s[3]
        
        visited_tiles = set()
        max_turn_count +=1 
        allowed_moves = []
        self.get_successors(s, allowed_moves, 0, chess_board, adv_pos, max_step, visited_tiles)

        print(f"allowed moves are {allowed_moves}")

        for i in range(len(allowed_moves)):
            successor = heapq.heappop(allowed_moves)
            print(f"successor is {successor}")
            alpha = max(alpha, self.min_value(successor, chess_board, alpha, beta, adv_pos, max_step, max_turn_count))
            if alpha >= beta: return alpha, s[1], s[2], s[3]

        print(f"return from max_value is {alpha, s[1], s[2], s[3]}")
        
        return alpha, s[1], s[2], s[3]
    
    def min_value(self, s, chess_board, alpha, beta, adv_pos, max_step, max_turn_count):
        allowed_moves = []
        visited_tiles = set()
        self.get_successors(s, allowed_moves, 0, chess_board, adv_pos, max_step, visited_tiles)
        print("-----------")
        print("in min_value")
        print(f"s is {s}")
        print("-----------")
        for i in range(len(allowed_moves)):
            successor = heapq.heappop(allowed_moves)
            max_turn = self.max_value(successor, chess_board, alpha, beta, adv_pos, max_step, max_turn_count)
            beta = min(beta, max_turn[0])
            if alpha >= beta: return alpha
        return beta
    
    # Taken from world.py
    def set_barrier(self, r, c, chess_board, dir):
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    def endgame(self, chess_board, my_pos, adv_pos):
        board_size = chess_board.shape[0]
        r,c = my_pos
        # Union-Find
        parent = dict()
        for r in range(board_size):
            for c in range(board_size):
                # self loop: initially, each element points to itseldf
                parent[(r,c)]= (r,c)

        def union(pos1, pos2):
            parent[pos1]= pos2

        # follow parent node until you reach a self loop
        def find(pos):
            if parent[pos] != pos:
                parent[pos] = find(parent[pos])
            return parent[pos]

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                        self.moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        my_pos_r = find(tuple(my_pos))
        adv_pos_r = find(tuple(adv_pos))
        my_pos_score = list(parent.values()).count(my_pos_r)
        adv_pos_score = list(parent.values()).count(adv_pos_r)
        if my_pos_r == adv_pos_r:
            return False, my_pos_score, adv_pos_score
        player_win = None
        win_blocks = -1
        if my_pos_score > adv_pos_score:
            player_win = 0
            win_blocks = my_pos_score
        elif my_pos_score < adv_pos_score:
            player_win = 1
            win_blocks = adv_pos_score
        else:
            player_win = -1  # Tie
        if player_win >= 0:
            logging.info(
                f"Game ends! Player {self.player_names[player_win]} wins having control over {win_blocks} blocks!"
            )
        else:
            logging.info("Game ends! It is a Tie!")
        return True, my_pos_score, adv_pos_score
    
    def state_of_game(chess_board):
        remaining_pos = []
        board_size = chess_board.shape[0]

        for i in range(board_size):
            for j in range(board_size):
                remaining_pos.extend([
                    d for d in range(4)  # 4 moves possible
                    if not chess_board[i, j, d]  # chess_board True means wall
                ])
        cur_count = len(remaining_pos)
        initial_count = (4 * board_size * board_size) - (4 * board_size)

        if cur_count / initial_count >= 0.4:
            return 1 #beginning game
        elif 0.4 < cur_count / initial_count <= 0.6:
            return 2 #middle game
        else:
            return 3 #end game

    def eval_function(self,chess_board, r, c, max_step, move_number):
        #where we return the complete heuristic, other defined heuristics should be called here and added to the return value
        available_move_number = self.get_available_move_number(chess_board, r, c, max_step, move_number)

        return available_move_number

    def get_available_move_number(self, chess_board, my_pos, adv_pos, max_step):
        #returns the number of available complete moves, not the number of available tiles
        allowed_moves = {}

        self.get_successors(allowed_moves, 0, chess_board, my_pos, adv_pos, max_step, -1)
        number_of_moves = 0

        for move in allowed_moves:
            number_of_moves+=len(allowed_moves[move])
        
        return number_of_moves
    
    def is_aggressive_move (self, my_pos, my_dir, adv_pos):
        my_r, my_c = my_pos
        adv_r, adv_c = adv_pos

        if (my_r == adv_r+1 and my_c == adv_c and my_dir == 0) or \
            (my_r == adv_r and my_c == adv_c-1 and my_dir == 1) or \
            (my_r == adv_r-1 and my_c == adv_c and my_dir == 0) or \
            (my_r == adv_r+1 and my_c == adv_c and my_dir == 0):
            return self.aggressive_heuristic
        
        return 0
    
    def distance_between_agents(my_pos, adv_pos):
        #the more the distance the less the heuristic value
        my_r, my_c = my_pos
        adv_r, adv_c = adv_pos

        return 1/(abs(my_r - adv_r) + abs(my_c - adv_c))

        

        






        
    

    
    
    









        
        
        



