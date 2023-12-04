# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import heapq
import logging
import random


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

        #allowed_moves = {}
       # self.get_successors(allowed_moves, 0,chess_board,my_pos, adv_pos, max_step, -1)
        #print(allowed_moves.keys())

        #if len(list(allowed_moves.keys())) == 0:
         #   r,c = my_pos
         #   dir = random.randint(0, 3)
         #   return (r,c), dir
        #allowed_pos = list(allowed_moves.keys())[0]
        #r,c = allowed_pos
        #move = r,c, allowed_moves[allowed_pos][0]
        
        #print("get_successor took ", time_taken_get_successpr, "seconds.")

    
        alpha, move = self.alpha_beta(my_pos, adv_pos, chess_board, max_step,)
        next_r,next_c, next_dir = move

        if(alpha>=1000):
            print("found winning move")

        #print(f"chosen move has alpha = {alpha}")
        #print(move)
        time_taken = time.time() - start_time
        
        #print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return (next_r,next_c), next_dir
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
            if not to_skip_pos:
                # adding the current tile and the possible directions for the wall on the dict of possible moves
                if (r,c) in allowed_moves:
                    if len(allowed_moves[(r,c)]) != len(allowed_dirs):
                        allowed_moves[(r,c)].append(dir)
                else:
                    allowed_moves[(r,c)] = [dir]

            #actually making the move and recursion for the remaining steps
            #when making the next move, make sure you're not moving to the direction agent was coming from 
            if dir!=prev_dir:
                m_r, m_c = self.moves[dir]
                next_pos = (r + m_r, c + m_c)
                self.get_successors(allowed_moves, i+1, chess_board,next_pos, adv_pos, max_step, dir)

    
    def alpha_beta(self, my_pos, adv_pos, chess_board, max_step):
        my_r, my_c = my_pos
        adv_r, adv_c = adv_pos
        return self.alpha_beta_max((my_r, my_c, -1), (adv_r, adv_c, -1), chess_board, float("-inf"), float("inf"), max_step, 0, 3)


    def alpha_beta_max(self, curr_move, adv_move, chess_board, alpha, beta, max_step, curr_depth, max_depth):
        #print("----")
        #print("Max node at depth", curr_depth)
        #print("Current state:", curr_move)
        #print("Alpha:", alpha, "Beta:", beta)
        #print("----")

        # CUTOFF
        is_cutoff, score = self.cutoff(curr_move, adv_move, chess_board, max_step, curr_depth, max_depth, True)
        if is_cutoff:
            #if curr_depth==0:
            #    print(f"returning score in first layer= {score}")
            return score, None
        
        eval = float('-inf')
        new_alpha = alpha

        allowed_moves = {}
        before_get_successor_time = time.time()
        r,c,dir = curr_move
        adv_r, adv_c, d = adv_move
        self.get_successors(allowed_moves, 0,chess_board,(r,c), (adv_r,adv_c), max_step, -1)
        print(allowed_moves.keys())
        after_get_successor_time = time.time()
        time_taken_get_successpr = after_get_successor_time - before_get_successor_time
        
        #print("get_successor took ", time_taken_get_successpr, "seconds.")

        allowed_moves_list = self.dict_to_heap(allowed_moves,curr_move, adv_move, chess_board, max_step, True)

        move_alpha_dict = {}

        for i in range(len(allowed_moves_list)):
            successor_w_h = heapq.heappop(allowed_moves_list)
            h, successor = successor_w_h

            copied_chess_board = deepcopy(chess_board)
            self.set_barrier(successor, copied_chess_board)

            min_return, new_move= self.alpha_beta_min(adv_move, successor, copied_chess_board, new_alpha, beta, max_step, curr_depth+1, max_depth)
            move_alpha_dict[successor] = min_return
            #alpha = max(alpha, min_return)

            #rewrite this as eval = max(min_return, eval)
            if min_return > eval:
                eval = min_return

            if eval >= beta:
                #print("pruned")
                ##if curr_depth==1:
                 #   print(f"returning score in first layer= {beta}")
                return eval, successor
            
            new_alpha = eval if eval > alpha else new_alpha

        #move_of_the_winning_alpha = curr_move
        for move in move_alpha_dict:
            if (move_alpha_dict[move] == eval):
                move_of_the_winning_alpha = move
                return eval, move_of_the_winning_alpha
            
        return(eval, None)

        #print("Returning from Max node with alpha =", alpha)
        #print("Returning from Max node with move =", move_of_the_winning_alpha)
        #if curr_depth==1:
        #    print(f"returning score in first layer= {alpha}")
        #return alpha, move_of_the_winning_alpha


    def alpha_beta_min(self, curr_move, adv_move, chess_board, alpha, beta, max_step, curr_depth, max_depth):
        #print("----")
        #print("Min node at depth", curr_depth)
        #print("Current state:", curr_move)
        #print("Alpha:", alpha, "Beta:", beta)
        #print("----")

        # CUTOFF
        is_cutoff, score = self.cutoff(adv_move, curr_move, chess_board, max_step, curr_depth, max_depth, False)
        if is_cutoff:
            return score, None
        
        eval = float('inf')
        new_beta = beta

        allowed_moves = {}
        r,c,dir = curr_move
        adv_r, adv_c, d = adv_move
        self.get_successors(allowed_moves, 0,chess_board,(r,c), (adv_r,adv_c), max_step, -1)
        print(allowed_moves.keys())
        allowed_moves_list = self.dict_to_heap(allowed_moves,curr_move, adv_move, chess_board, max_step, False)

        move_beta_dict = {}

        for i in range(len(allowed_moves_list)):
            successor_w_h = heapq.heappop(allowed_moves_list)
            h, successor = successor_w_h

            copied_chess_board = deepcopy(chess_board)
            self.set_barrier(successor, copied_chess_board)

            max_return, new_move = self.alpha_beta_max(adv_move, successor, chess_board, alpha, new_beta, max_step, curr_depth+1, max_depth)
            move_beta_dict[successor] = new_beta
            #beta = min(beta, new_beta)

            if max_return < eval:
                eval= max_return

            if alpha >= eval:
                #print("pruned")
                return eval, successor
            
            new_beta = eval if eval < new_beta else new_beta
        
            
        return(eval, None)
            
        #print("Returning from Min node with alpha =", alpha)
        #print("Returning from Min node with move =", move_of_the_winning_beta)
        #return beta, move_of_the_winning_beta
    
    def cutoff(self,curr_move, adv_move, chess_board, max_step, curr_depth, max_depth, is_max):
        #returns a boolean that indicates whether we're at a cutoff and a float indicating our score or the eval function's result
        r,c,dir = curr_move
        adv_r, adv_c, d = adv_move
        is_endgame, score = self.endgame(chess_board, (r,c), (adv_r, adv_c))
        
        
        if is_endgame:
            #print("game ended")
            #print(f"is_max: {is_max}")
            #print(curr_move)
            #print(score)
            if score == 0 :
                #tie
                return True, 0
            return True, score*100
        
        elif curr_depth == max_depth:
            return True, self.eval_function(curr_move, adv_move, chess_board, max_step, is_max)
        return False, 0

    # Taken from world.py
    def set_barrier(self, curr_move,  chess_board):
        # Set the barrier to True
        r,c,dir = curr_move
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
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        my_pos_r = find(tuple(my_pos))
        adv_pos_r = find(tuple(adv_pos))
        my_pos_score = list(parent.values()).count(my_pos_r)
        adv_pos_score = list(parent.values()).count(adv_pos_r)
        if my_pos_r == adv_pos_r:
            return False, my_pos_score - adv_pos_score
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
        
        return True, my_pos_score - adv_pos_score
    
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

    def eval_function(self,curr_move, adv_move, chess_board, max_step, is_max):
        #where we return the complete heuristic, other defined heuristics should be called here and added to the return value
        r,c,dir = curr_move
        adv_r, adv_c, d = adv_move
        available_move_number = self.get_number_of_reachable_tiles(chess_board, (r,c), (adv_r, adv_c), max_step, is_max)
        aggression_heuristic = self.is_aggressive_move(curr_move, (adv_r, adv_c))
        #distance_heuristic = self.distance_between_agents((r,c), (adv_r, adv_c))/10
        winning_heuristic = self.winning_heuristic((r,c), (adv_r, adv_c), chess_board, max_step)

        op_available_move_number = self.get_number_of_reachable_tiles(chess_board, (adv_r, adv_c),(r,c), max_step, is_max)
        op_aggression_heuristic = self.is_aggressive_move(adv_move, (r,c))
        #op_distance_heuristic = self.distance_between_agents((adv_r, adv_c),(r,c))/10
        op_winning_heuristic = self.winning_heuristic((adv_r, adv_c), (r,c), chess_board, max_step)
        #if (not is_max):
        #    return available_move_number
        #return -available_move_number
        return available_move_number + aggression_heuristic +winning_heuristic - (op_available_move_number+op_aggression_heuristic)
    

    def get_number_of_reachable_tiles(self, chess_board, my_pos, adv_pos, max_step, is_max):
        #returns the number of available complete moves, not the number of available tiles
        allowed_moves = {}

        self.get_successors(allowed_moves, 0, chess_board, my_pos, adv_pos, max_step, -1)
        #if(not is_max):
            #print(len(list(allowed_moves.keys())))
        return len(list(allowed_moves.keys()))

    
    
    def is_aggressive_move (self, my_move, adv_pos):
        my_r, my_c, my_dir = my_move
        adv_r, adv_c = adv_pos

        if (my_r == adv_r+1 and my_c == adv_c and my_dir == 0) or \
            (my_r == adv_r and my_c == adv_c-1 and my_dir == 1) or \
            (my_r == adv_r-1 and my_c == adv_c and my_dir == 0) or \
            (my_r == adv_r+1 and my_c == adv_c and my_dir == 0):
            return self.aggression_heuristic
        
        return 0
    
    def distance_between_agents(self, my_pos, adv_pos):
        #the more the distance the less the heuristic value
        my_r, my_c = my_pos
        adv_r, adv_c = adv_pos

        return 10/(abs(my_r - adv_r) + abs(my_c - adv_c))
    
    def winning_heuristic(self, my_pos, adv_pos, board, max_step):
        #if the adversary has three sides blocked with walls and if we're able to move there we should trap it
        
        adv_r, adv_c = adv_pos
        allowed_dirs = [ d                                
            for d in range(0,4)                           # 4 moves possible
            if not board[adv_r,adv_c,d] and                 # chess_board True means wall
            not my_pos == (adv_r+self.moves[d][0],adv_c+self.moves[d][1])] # cannot move through Adversary

        if len(allowed_dirs) == 1:
            #adversary has 3 sides blocked

            #if I'm not next to the adversary

            if self.distance_between_agents(my_pos, adv_pos)<max_step:
                dir = allowed_dirs[0]
                if my_pos == (adv_r+self.moves[dir][0],adv_c+self.moves[dir][1]):
                    return 100
        return 0

        
    def dict_to_heap(self, dict, curr_move, adv_move, chess_board, max_step, is_max):
        allowed_moves_list = []
        for pos in dict:
            for dir in dict[pos]:
                r,c = pos
                eval = self.eval_function(curr_move, adv_move, chess_board, max_step, is_max)
                heapq.heappush(allowed_moves_list, (eval, (r,c,dir)))
        return allowed_moves_list

        






        
    

    
    
    









        
        
        



