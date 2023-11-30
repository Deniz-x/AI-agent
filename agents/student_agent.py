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
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

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
        print(f"starting s is {(0,r,c,-1)}")
        #self.get_successors(allowed_moves, 0, chess_board,my_pos, adv_pos, max_step, -1)
        next_move = self.max_value((0,r,c,-1), chess_board, float('-inf'), float('inf'), adv_pos, max_step, 0)
        print(f"next move is {next_move}")
        heuristic, row, column, dir = next_move
        next_position = row,column

        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return next_position, dir
    
    #my_pos, self.dir_map["u"]

    #self.dir_map["u"]

    def get_successors(self, s, allowed_moves, i,chess_board, adv_pos, max_step):

        
        heuristic, r, c, prev_dir = s
        
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
            not adv_pos == (r+moves[d][0],c+moves[d][1]) and
            d!=prev_dir] # cannot move through Adversary
        
        
        if len(allowed_dirs)==0:
            # If no possible move, we must be enclosed by our Adversary
            return 0
        
        if len(allowed_dirs)==1: #&& i!=1:
            return -1
        
        heuristic = np.random.randint(0, len(chess_board))
        
        for dir in allowed_dirs:
            
            #allowed_moves.append((r,c,dir))
            m_r, m_c = moves[dir]
            next_s = (heuristic, r + m_r, c + m_c, dir)
            is_good_move = self.get_successors(next_s, allowed_moves, i+1, chess_board, adv_pos, max_step)
            if is_good_move != -1:
                heapq.heappush(allowed_moves, (-heuristic, r,c,dir))

    #def alpha_beta():


    def max_value(self, s, chess_board, alpha, beta, adv_pos, max_step, max_turn_count):
        print("in max_value")
        print(f"s is {s}")

        if max_turn_count==max_depth:
            return -s[0],s[1],s[2], s[3]
        max_turn_count +=1 
        allowed_moves = []
        self.get_successors(s, allowed_moves, 0, chess_board, adv_pos, max_step)

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
        self.get_successors(s, allowed_moves, 0, chess_board, adv_pos, max_step)
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
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

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

    def eval_function(self):
        self.get_available_move_number()

    #def get_available_move_number(self, chess_board, r, c, max_step, move number):
        






        
    

    
    
    









        
        
        



