
from src.BackgammonMove import BackGammonMoveBlack, BackgammenMoveWhite, BackgammonMove
from src.BackgammonState import BackgammonState
from src.constants import LAST_INDEX_FIELD_BOARD, FIRST_INDEX_FIELD_BOARD
import copy
from src.BackgammenInvariants import backgammonstate_invariant


"""
     Exposing API should include:
     
"""



def generate_moves(is_black : bool) -> list[BackgammonMove]:
     pass


def _generate_black_moves(board_state : BackgammonState) -> list[BackGammonMoveBlack]:
     pass

def _generate_white_moves(board_state : BackgammonState) -> list[BackgammenMoveWhite]:
     pass


def _generate_black_game_states(board_state : BackgammonState, dice : list[int]) -> list[BackgammonState]:

     all_poss_states : list[BackgammonState] = []

     def backtrack_states(inner_board_state : BackgammonState , inner_dice : list[int]) -> None:
          if len(inner_dice) == 0:
               all_poss_states.append(copy.deepcopy(inner_board_state))
               return
          
          if inner_board_state.blackBearing:
              # here still the opportunity is to hit a white stone..
               poss_moves = _beat_move_possible_in_bearing_off(game_state=inner_board_state, dice=inner_dice[0])
               if poss_moves is not None:
                    for m in poss_moves:
                         if _valid_move_black(game_state=inner_board_state, move_black=m):
                              backtrack_states(update_board_move_black(inner_board_state , m), inner_dice=inner_dice[1:])

               backtrack_states(_update_board_black_bearing(inner_board_state , inner_dice[0]), inner_dice=inner_dice[1:])

          # just the normal case
          else:
               for x in range(len(inner_board_state.board)):

                    if inner_board_state.board[x] > 0:

                         poss_move = BackGammonMoveBlack(fromField=x , toField= x + inner_dice[0])
                         if _valid_move_black(inner_board_state, poss_move):
                              backtrack_states(update_board_move_black(inner_board_state , poss_move), inner_dice=inner_dice[1:])
          
     backtrack_states(inner_board_state=board_state , inner_dice=dice)

     return all_poss_states

"""
     there might be the possibility to beat a white checke while in the bearing of phase
"""
def _beat_move_possible_in_bearing_off(game_state, dice : int) -> list[BackGammonMoveBlack] | None:
     beat_moves : list[BackgammonMove] = [] 
     for idx in range(18, 24, 1):
          if  idx + dice <= 23 and  game_state.board[idx + dice] == 1:
               beat_moves.append(BackGammonMoveBlack(idx, toField=idx + dice))
     
     return None

"""
     this takes a backgammon state and bears a stone off given these rules.

          the goal is to virtually go to 24 for black.
          we can take a stone of on 24 - x ==> target field
          should there be no stone on this field. we go back until field with index is is reached

          otherwise we move forward from the target field and 


"""
def _update_board_black_bearing(game_state : BackgammonState, dice : int) -> BackgammonState:
     new_game_state = copy.deepcopy(game_state)

     VIRTUAL_INDEX_TO_REACH = 24
     FIELD_18 = 18
     LAST_FIELD = 23

     target_field_index = VIRTUAL_INDEX_TO_REACH - dice
     target_field = new_game_state.board[target_field_index]
     if target_field > 0:
          # bear of the stone
          new_game_state.board[target_field_index] -= 1
          new_game_state.blackOutside += 1
          return new_game_state
     
     if target_field == 0:
          # first move upwards
          for idx in range(target_field -1 , FIELD_18 -1, -1):
               if new_game_state.board[idx] > 0:
                    # bear of the checker
                    new_game_state.board[target_field_index] -= 1
                    new_game_state.blackOutside += 1
                    return new_game_state
          
          for idx in range(target_field +1 , LAST_FIELD +1, 1): 
               if new_game_state.board[idx] > 0:
                    new_game_state.board[target_field_index] -= 1
                    new_game_state.blackOutside += 1
                    return new_game_state


"""
     valid move for black ( going left to right) :
          if on the fromField there is at least one black stone
               AND
          (the to_field is empty 
               OR 
          has exactly one white stone 
               OR
          has an arbitrary number of black stones)

               AND

          it should be generally in bounds

"""
def _valid_move_black(game_state : BackgammonState, move_black : BackGammonMoveBlack) -> bool:
     if game_state.blackBearing:
          return _move_in_bounds_bearing_black(move=move_black)
     else:
          return _move_in_bounds_no_bearing(move=move_black)


def _move_in_bounds_no_bearing(move : BackgammonMove) -> bool:
     return (move.fromField >= FIRST_INDEX_FIELD_BOARD and move.fromField <= LAST_INDEX_FIELD_BOARD) and (move.toField >= FIRST_INDEX_FIELD_BOARD and move.toField <= LAST_INDEX_FIELD_BOARD)


def _move_in_bounds_bearing_black(move : BackGammonMoveBlack) -> bool:
     return (move.fromField >= 18 and move.toField <=24)

"""
     This function applies exactly one mve to the board.

"""
def update_board_move_black(game_state : BackgammonState, move_black : BackGammonMoveBlack) -> BackgammonState:

     new_board_state : BackgammonState = copy.deepcopy(game_state)

     # a black checker gets inserted
     if move_black.fromField == 0 and move_black.insertionMove:
          new_board_state.blackCaught -= 1
          new_board_state.board[move_black.toField] += 1
          new_board_state.blackBearing = is_black_bearing(new_board_state)
          backgammonstate_invariant(new_board_state)
          return new_board_state

     new_board_state.board[move_black.fromField] -= 1

     # in case a white checker gets eaton
     if new_board_state.board[move_black.toField] == -1:
          new_board_state.whiteCaught += 1
          new_board_state.board[move_black.toField] = 1
          new_board_state.blackBearing = is_black_bearing(new_board_state)
          backgammonstate_invariant(new_board_state)
          return new_board_state
     
     # the normal case: just a move
     if new_board_state.board[move_black.toField] >= 0:
          new_board_state.board[move_black.toField] += 1

     
     new_board_state.blackBearing = is_black_bearing(new_board_state)
     backgammonstate_invariant(new_board_state)
     return new_board_state

"""
     Checks the fields 18 till 23 if there are 15 checkers in there
"""
def is_black_bearing(game_state : BackgammonState) -> bool:
     if game_state.blackCaught > 0:
          return False
     
     board_slice = game_state.board[17:24]
     sum = 0
     for x in board_slice:
          if x > 0:
               sum += x
     
     return (sum + game_state.blackOutside) == 15








          



