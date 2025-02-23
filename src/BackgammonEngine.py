
from BackgammonMove import BackGammonMoveBlack, BackgammenMoveWhite, BackgammonMove
from BackgammonState import BackgammonState
import copy


"""
     Exposing API should include:
     
     All possible states at timeszep t:
          generate_moves(is_black : bool)

     Take action a_t at state s_t:
      
          Black version:
               update_board_with_move_black(board, move) -> board

          White version:
               update_board_with_move_white(board, move) -> board
"""



"""
     implements the function : A(s_t)
          gives back a set of all possible Moves for a given state S at timestep t
     
"""

def generate_moves(is_black : bool) -> list[BackgammonMove]:
     pass


def _generate_black_moves(board_state : BackgammonState) -> list[BackGammonMoveBlack]:
     pass

def _generate_white_moves(board_state : BackgammonState) -> list[BackgammenMoveWhite]:
     pass


def _generate_black_moves_normal_case(board_state : BackgammonState, dice : list[int]) -> list[BackGammonMoveBlack]:
     all_poss_moves : list[BackGammonMoveBlack] = []

     def backtrack_moves(inner_board_state : BackgammonState , dice : list[int]) -> None:
          pass

"""
     valid move for black ( going left to right) :
          if on the fromField there is at least one black stone
               AND
          (the to_field is empty 
               OR 
          has exactly one white stone 
               OR
          has an arbitrary number of black stones)
"""
def _valid_move_black(game_state : BackgammonState, move_black : BackGammonMoveBlack) -> bool:
     return game_state.board[move_black.fromField] > 0 and (game_state.board[move_black.toField] == -1 or
                          game_state.board[move_black.toField] == 0 or game_state.board[move_black.toField > 0])



"""
     This function applies a tuple of moves on a board for black.
          a tuple consists of 2 or 4 moves (2 or dice)

"""
def update_board_move_black(game_state : BackgammonState, moves_black : list[BackGammonMoveBlack]) -> BackgammonState:

     new_board_state : BackgammonState = copy.deepcopy(game_state)

     for move in moves_black:
          pass



          



