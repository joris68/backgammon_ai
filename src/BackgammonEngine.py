
from src.BackgammonMove import BackGammonMoveBlack, BackgammenMoveWhite, BackgammonMove
from src.BackgammonState import BackgammonState
from src.constants import LAST_INDEX_FIELD_BOARD, FIRST_INDEX_FIELD_BOARD
import copy
from src.BackgammenInvariants import backgammonstate_invariant
from collections import Counter
import logging

logger = logging.getLogger(__name__)

"""
     Exposing API should include:
     
"""

def get_unused_items(all_items, used_items):
    all_count = Counter(all_items)
    used_count = Counter(used_items)
    for item, count in used_count.items():
        all_count[item] -= count

    unused_items = []
    for item, count in all_count.items():
        unused_items.extend([item] * max(count, 0))

    return unused_items

def generate_moves(game_state : BackgammonState, is_black : bool , dice : list[int]) -> list[BackgammonState]:
     if is_black:
          return _generate_black_moves(game_state=game_state, dice= dice)
     else:
          return _generate_white_moves(game_state=game_state, dice=dice)


def _generate_black_moves(game_state : BackgammonState, dice : list[int]) -> list[BackgammonState]:
     
     dice_used = []

     if game_state.blackCaught > 0:
          game_state , dice_used = _insert_stones_black(game_state=game_state, dice=dice)
     
     if len(dice_used) == len(dice) or (len(dice_used) == 0 and game_state.blackCaught > 0):
          return [game_state] 

     return _generate_black_game_states(board_state=game_state, dice=get_unused_items(dice, dice_used))

def _generate_white_moves(game_state : BackgammonState, dice : list[int]) -> list[BackgammonState]:

     dice_used = []

     if game_state.whiteCaught > 0:
          game_state , dice_used = _insert_stones_white(game_state=game_state, dice=dice)
     
     if len(dice_used) == len(dice) or (len(dice_used) == 0 and game_state.whiteCaught > 0):
          return [game_state]

     return _generate_white_game_states(board_state=game_state, dice=get_unused_items(dice, dice_used))
   

"""
     gives back the end game state and the number of insertion done. Inserton black done from [0, 5]
"""
def _insert_stones_black(game_state : BackgammonState , dice : list[int]) -> tuple[BackgammonState , list[int], int]:
     dice_used = []
     for d in dice:
          if  game_state.blackCaught > 0 and game_state.board[d] >= 0:
               game_state = update_board_move_black(game_state=game_state, move_black=BackGammonMoveBlack(fromField=-1 , toField=d , insertionMove=True,  bearingOffMove=False))
               dice_used.append(d)
     
     return game_state, dice_used 


def _insert_stones_white(game_state : BackgammonState , dice : list[int]) -> tuple[BackgammonState , list[int]]:
     dice_used = []
     for d in dice:
          if  game_state.whiteCaught > 0 and game_state.board[24 - d] <= 0:
               game_state = update_board_move_white(game_state=game_state, move_white=BackgammenMoveWhite(fromField=24 , toField=(24 - d) , insertionMove=True,  bearingOffMove=False))
               dice_used.append(d)
     
     return game_state, dice_used



def _generate_black_game_states(board_state : BackgammonState, dice : list[int]) -> list[BackgammonState]:
     all_poss_states : set[BackgammonState] = set()

     def backtrack_states(inner_board_state : BackgammonState , inner_dice : list[int]) -> None:
          if inner_board_state.ended:
               if inner_board_state not in all_poss_states:
                    all_poss_states.add(copy.deepcopy(inner_board_state))
               return

          if len(inner_dice) == 0:
               if inner_board_state not in all_poss_states:
                    all_poss_states.add(copy.deepcopy(inner_board_state))
               return
          
          if inner_board_state.blackBearing:
              # here still the opportunity is to hit a white stone..
               poss_moves = _beat_move_possible_in_bearing_off_black(game_state=inner_board_state, dice=inner_dice[0])
               if poss_moves is not None:
                    for m in poss_moves:
                         if _valid_move_black(game_state=inner_board_state, move_black=m):
                              backtrack_states(update_board_move_black(inner_board_state , m), inner_dice=copy.deepcopy(inner_dice[1:]))

               backtrack_states(_update_board_black_bearing(inner_board_state , inner_dice[0]), inner_dice=copy.deepcopy(inner_dice[1:]))

          # just the normal case
          else:
               for x in range(len(inner_board_state.board)):

                    if inner_board_state.board[x] > 0:

                         poss_move = BackGammonMoveBlack(fromField=x , toField= x + inner_dice[0])
                         if _valid_move_black(inner_board_state, poss_move):
                              backtrack_states(update_board_move_black(inner_board_state , poss_move), inner_dice=copy.deepcopy(inner_dice[1:]))
          
     backtrack_states(inner_board_state=board_state , inner_dice=dice)

     if len(all_poss_states) == 0:
          return [board_state]
     else:
          return list(all_poss_states)

"""
     there might be the possibility to beat a white checke while in the bearing of phase
"""
def _beat_move_possible_in_bearing_off_black(game_state, dice : int) -> list[BackGammonMoveBlack] | None:
     beat_moves : list[BackgammonMove] = [] 
     for idx in range(18, 24, 1):
          if  idx + dice <= 23 and  game_state.board[idx + dice] == -1 and game_state.board[idx] > 0:
               beat_moves.append(BackGammonMoveBlack(idx, toField=idx + dice))
     
     if len(beat_moves) > 0:
          return beat_moves
     else:
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
          new_game_state.ended = _game_ended_black(new_game_state)
          backgammonstate_invariant(game_state_before=game_state, game_state_after=new_game_state)
          return new_game_state
     
     # first move upwards
     for idx in range(FIELD_18 , target_field_index, 1):
          if new_game_state.board[idx] > 0 and (new_game_state.board[idx + dice] >= 0 or new_game_state.board[idx + dice] == -1):
               # bear of the checker
               new_game_state.board[idx] -= 1
               if new_game_state.board[idx + dice] == -1:
                    new_game_state.board[idx + dice] = 1
                    new_game_state.whiteCaught += 1
               else:
                    new_game_state.board[idx + dice] += 1
               new_game_state.ended = _game_ended_black(new_game_state)
               backgammonstate_invariant(game_state_before=game_state, game_state_after=new_game_state)
               return new_game_state
          
     for idx in range(target_field_index +1 , LAST_FIELD +1, 1): 
          if new_game_state.board[idx] > 0:
               new_game_state.board[idx] -= 1
               new_game_state.blackOutside += 1
               new_game_state.ended = _game_ended_black(new_game_state)
               backgammonstate_invariant(game_state_before=game_state, game_state_after=new_game_state)
               return new_game_state

     return new_game_state
     #logger.info(f"gamestate : {new_game_state} dice : {dice}")          
     #raise Exception("something wrong in the blackBearing")


def _update_board_white_bearing(game_state : BackgammonState, dice : int) -> BackgammonState:
     new_game_state = copy.deepcopy(game_state)

     VIRTUAL_INDEX_TO_REACH = -1
     FIELD_5 = 5
     FIRST_FIELD = 0

     target_field_index = VIRTUAL_INDEX_TO_REACH + dice
     target_field = new_game_state.board[target_field_index]
     if target_field < 0:
          # bear of the stone
          new_game_state.board[target_field_index] += 1
          new_game_state.whiteOutside += 1
          new_game_state.ended = _game_ended_white(new_game_state)
          new_game_state.whiteBearing = is_white_bearing(new_game_state)
          new_game_state.blackBearing = is_black_bearing(new_game_state)
          backgammonstate_invariant(game_state_before=game_state, game_state_after=new_game_state)
          return new_game_state
     
     
     # first move upwards
     for idx in range(target_field_index +1 , FIELD_5 +1 , 1):
          if new_game_state.board[idx] < 0 and (new_game_state.board[idx - dice] == 1 or new_game_state.board[idx - dice] <= 0 ):
               new_game_state.board[idx] += 1
               if new_game_state.board[idx - dice] == 1:
                    new_game_state.board[idx - dice] = -1
                    new_game_state.blackCaught += 1
               else:
                    new_game_state.board[idx]  -= 1
               new_game_state.ended = _game_ended_white(new_game_state)
               new_game_state.whiteBearing = is_white_bearing(new_game_state)
               new_game_state.blackBearing = is_black_bearing(new_game_state)
               backgammonstate_invariant(game_state_before=game_state, game_state_after=new_game_state)
               return new_game_state
     #move downwards
     for idx in range(target_field_index - 1 , FIRST_FIELD -1, -1): 
          if new_game_state.board[idx] < 0:
               new_game_state.board[idx] += 1
               new_game_state.whiteOutside += 1
               new_game_state.ended = _game_ended_white(new_game_state)
               new_game_state.whiteBearing = is_white_bearing(new_game_state)
               new_game_state.blackBearing = is_black_bearing(new_game_state)
               backgammonstate_invariant(game_state_before=game_state, game_state_after=new_game_state)
               return new_game_state
     
     return new_game_state
     #logger.info(f"gamestate : {new_game_state} dice : {dice}")
     #raise Exception("something wrong in the whiteBearing")

def _game_ended_black(game_state : BackgammonState) -> bool:
     return game_state.blackOutside == 15

def _game_ended_white(game_state : BackgammonState) -> bool:
     return game_state.whiteOutside == 15

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
          return _move_in_bounds_bearing_black(move=move_black) and _moves_right_black(game_state=game_state, move=move_black)
     else:
          return _move_in_bounds_no_bearing(move=move_black) and _moves_right_black(game_state=game_state, move=move_black)


def _moves_right_black(game_state : BackgammonState, move : BackGammonMoveBlack) -> bool:
     return game_state.board[move.fromField] >= 1 and (game_state.board[move.toField] >= 1 or game_state.board[move.toField] == 0 or game_state.board[move.toField] == -1)

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
     if move_black.fromField == -1 and move_black.insertionMove:
          new_board_state.blackCaught -= 1
          new_board_state.board[move_black.toField] += 1
          new_board_state.blackBearing = is_black_bearing(new_board_state)
          new_board_state.whiteBearing = is_white_bearing(new_board_state)
          backgammonstate_invariant(game_state_before=game_state, game_state_after=new_board_state)
          return new_board_state

     new_board_state.board[move_black.fromField] -= 1

     # in case a white checker gets eaton
     if new_board_state.board[move_black.toField] == -1:
          new_board_state.whiteCaught += 1
          new_board_state.board[move_black.toField] = 1
          new_board_state.blackBearing = is_black_bearing(new_board_state)
          new_board_state.whiteBearing = is_white_bearing(new_board_state)
          backgammonstate_invariant(game_state_before=game_state, game_state_after=new_board_state)
          return new_board_state
     
     # the normal case: just a move
     if new_board_state.board[move_black.toField] >= 0:
          new_board_state.board[move_black.toField] += 1

     
     new_board_state.blackBearing = is_black_bearing(new_board_state)
     new_board_state.whiteBearing = is_white_bearing(new_board_state)
     backgammonstate_invariant(game_state_before=game_state, game_state_after=new_board_state)
     return new_board_state

"""
     Checks the fields 18 till 23 if there are 15 checkers in there
"""
def is_black_bearing(game_state : BackgammonState) -> bool:
     if game_state.blackCaught > 0:
          return False
     # 17 till 24
     board_slice = game_state.board[18:24]
     sum = 0
     for x in board_slice:
          if x > 0:
               sum += x
     
     return (sum + game_state.blackOutside) == 15


###################################################################

def _generate_white_game_states(board_state : BackgammonState, dice : list[int]) -> list[BackgammonState]:

     all_poss_states : set[BackgammonState] = set()

     def backtrack_states(inner_board_state : BackgammonState , inner_dice : list[int]) -> None:
          if inner_board_state.ended:
               if inner_board_state not in all_poss_states:
                    all_poss_states.add(copy.deepcopy(inner_board_state))
               return

          if len(inner_dice) == 0:
               if inner_board_state not in all_poss_states:
                    all_poss_states.add(copy.deepcopy(inner_board_state))
               return
          
          if inner_board_state.whiteBearing:
              # here still the opportunity is to hit a black stone..
               poss_moves = _beat_move_possible_in_bearing_off_white(game_state=inner_board_state, dice=inner_dice[0])
               if poss_moves is not None:
                    for m in poss_moves:
                         if _valid_move_white(game_state=inner_board_state, move_white=m):
                              backtrack_states(update_board_move_white(inner_board_state , m), inner_dice=copy.deepcopy(inner_dice[1:]))

               backtrack_states(_update_board_white_bearing(inner_board_state , inner_dice[0]), inner_dice=copy.deepcopy(inner_dice[1:]))

          # just the normal case
          else:
               for x in range(len(inner_board_state.board)):

                    if inner_board_state.board[x] < 0:

                         poss_move = BackgammenMoveWhite(fromField=x , toField= x - inner_dice[0])
                         if _valid_move_white(inner_board_state, poss_move):
                              backtrack_states(update_board_move_white(inner_board_state , poss_move), inner_dice=copy.deepcopy(inner_dice[1:]))
          
     backtrack_states(inner_board_state=board_state , inner_dice=dice)

     if len(all_poss_states) == 0:
          return [board_state]
     else:
          return list(all_poss_states)


"""
     there might be the possibility to beat a black checkers while in the bearing of phase
"""
def _beat_move_possible_in_bearing_off_white(game_state, dice : int) -> list[BackgammenMoveWhite] | None:
     beat_moves : list[BackgammonMove] = [] 
     for idx in range(5, -1, -1):
          if game_state.board[idx] < 0 and  idx - dice >= 0 and  game_state.board[idx - dice] == 1:
               beat_moves.append(BackgammenMoveWhite(idx, toField=idx - dice))
     
     if len(beat_moves) > 0:
          return beat_moves
     else:
          return None

def _valid_move_white(game_state : BackgammonState, move_white : BackgammenMoveWhite) -> bool:

     if game_state.whiteBearing:
          return _move_in_bounds_bearing_white(move=move_white)
     else:
           return _move_in_bounds_no_bearing(move=move_white) and _moves_right_white(game_state=game_state, move=move_white)

def _moves_right_white(game_state : BackgammonState, move : BackgammenMoveWhite) -> bool:
     return game_state.board[move.fromField] <= -1 and (game_state.board[move.toField] <= -1 or game_state.board[move.toField] == 0 or game_state.board[move.toField] == 1)

    
def _move_in_bounds_bearing_white(move : BackgammenMoveWhite) -> bool:
     return (move.fromField <= 5 and move.toField >= -1)


def update_board_move_white(game_state : BackgammonState, move_white : BackgammenMoveWhite) -> BackgammonState:

     new_board_state : BackgammonState = copy.deepcopy(game_state)

     # a white checker gets inserted
     if move_white.fromField == 24 and move_white.insertionMove:
          new_board_state.whiteCaught -= 1
          new_board_state.board[move_white.toField] -= 1
          new_board_state.whiteBearing = is_white_bearing(new_board_state)
          new_board_state.blackBearing = is_black_bearing(new_board_state)
          backgammonstate_invariant(game_state_before=game_state, game_state_after=new_board_state)
          return new_board_state
     
     new_board_state.board[move_white.fromField] += 1

     # in case a black checker gets eaton
     if new_board_state.board[move_white.toField] == 1:
          new_board_state.blackCaught += 1
          new_board_state.board[move_white.toField] = -1
          new_board_state.whiteBearing = is_white_bearing(new_board_state)
          new_board_state.blackBearing = is_black_bearing(new_board_state)
          backgammonstate_invariant(game_state_before=game_state, game_state_after=new_board_state)
          return new_board_state

     # the normal case: just a move
     if new_board_state.board[move_white.toField] <= 0:
          new_board_state.board[move_white.toField] -= 1

     new_board_state.whiteBearing = is_white_bearing(new_board_state)
     new_board_state.blackBearing = is_black_bearing(new_board_state)
     backgammonstate_invariant(game_state_before=game_state, game_state_after=new_board_state)
     return new_board_state
     

def is_white_bearing(game_state : BackgammonState) -> bool:
     if game_state.whiteCaught > 0:
          return False
     # 0 till 5 including or 6 excluding
     board_slice = game_state.board[0:6]
     sum = 0
     for x in board_slice:
          if x < 0:
               sum += abs(x)
     
     return (sum + game_state.whiteOutside) == 15
