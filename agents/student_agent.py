# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import heapq


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

        allowed_moves = []

        self.get_successors(allowed_moves, 0, chess_board,my_pos, adv_pos, max_step, -1)

        successors = []
        successors = list(set(allowed_moves))

        print(successors)

        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]

    #self.dir_map["u"]

    def get_successors(self, allowed_moves, i,chess_board,my_pos, adv_pos, max_step, prev_direction):

        
        #Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        
        #make sure we don't consider moving back
        if prev_direction!=-1:
            if prev_direction == 0:
                prev_direction = 2
            elif prev_direction == 1:
                prev_direction = 3
            elif prev_direction == 2:
                prev_direction = 0
            elif prev_direction == 3:
                prev_direction = 1

        #stop making moves when reached the max_step
        if i > max_step:
            return
        
        r, c = my_pos

        # Build a list of the moves we can make
        allowed_dirs = [ d                                
            for d in range(0,4)                           # 4 moves possible
            if not chess_board[r,c,d] and                 # chess_board True means wall
            not adv_pos == (r+moves[d][0],c+moves[d][1]) and
            d!=prev_direction] # cannot move through Adversary
        
        
        
        if len(allowed_dirs)==0:
            # If no possible move, we must be enclosed by our Adversary
            return 
        
        #if len(allowed_dirs)==1 and i!=0:
            # don't make the move that puts you in this position
        #    return
        
        for dir in allowed_dirs:

            allowed_moves.append((r,c,dir))
            m_r, m_c = moves[dir]
            next_pos = (r + m_r, c + m_c)
            self.get_successors(allowed_moves, i+1, chess_board,next_pos, adv_pos, max_step, dir)
            


        
        
        



