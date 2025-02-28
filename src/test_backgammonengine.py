import pytest
from src.utils import generate_dice_for_move
from src.BackgammonState import BackgammonState
from src.BackgammonMove import BackGammonMoveBlack
from src.BackgammonEngine import  _generate_black_game_states


def get_starting_state() -> BackgammonState:
     return BackgammonState([2, 0, 0, 0, 0, -5,   0, -3, 0 ,0 ,0 , 5,   -5, 0,0,0,3,0,   5, 0,0,0,0,-2], whiteCaught=0, blackCaught=0,
                     blackBearing=False, whiteBearing=False, blackOutside=0, whiteOutside=0, ended=False)

def one_step_before_bearing_off() -> BackgammonState:
     return BackgammonState([0, 0, 0, 0, -1, -5,   0, -3, 0 ,0 ,0 , 0,   -5, 0,0,0,1,0,   5, 3,5,1,-1,0], whiteCaught=0, blackCaught=0,
                     blackBearing=False, whiteBearing=False, blackOutside=0, whiteOutside=0, ended=False)

"""
def test_generate_normal_black_1():
     game_state = get_starting_state()
     moves = _generate_black_moves_normal_case(game_state, [2])
     #print(moves)

def test_generate_normal_black_2():
     game_state = get_starting_state()
     moves = _generate_black_moves_normal_case(game_state, [2, 3])
     #print(moves)

def test_generate_normal_black_3():
     game_state = get_starting_state()
     moves = _generate_black_game_states(game_state, [2])
     #print(moves)
    # print(moves)
"""
def test_one_before_bearing_move():
     game_state = one_step_before_bearing_off()
     moves = _generate_black_game_states(game_state, [3, 3])
     print(moves)


