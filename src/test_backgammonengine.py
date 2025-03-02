import pytest
from src.utils import generate_dice_for_move
from src.BackgammonState import BackgammonState
from src.BackgammonMove import BackGammonMoveBlack
from src.BackgammonEngine import  generate_moves, get_unused_items


def get_starting_state() -> BackgammonState:
     return BackgammonState([2, 0, 0, 0, 0, -5,   0, -3, 0 ,0 ,0 , 5,   -5, 0,0,0,3,0,   5, 0,0,0,0,-2], whiteCaught=0, blackCaught=0,
                     blackBearing=False, whiteBearing=False, blackOutside=0, whiteOutside=0, ended=False)

def one_step_before_bearing_off() -> BackgammonState:
     return BackgammonState([0, 0, 0, 0, -1, -5,   0, -3, 0 ,0 ,0 , 0,   -5, 0,0,0,1,0,   5, 3,5,1,-1,0], whiteCaught=0, blackCaught=0,
                     blackBearing=False, whiteBearing=False, blackOutside=0, whiteOutside=0, ended=False)


def white_insertion() -> BackgammonState:
     return BackgammonState([0, 0, 0, 0, -1, -5,   0, -3, 0 ,0 ,0 , 0,   -5, 0,0,0,1,0,   5, 3,5,0,-1,0], whiteCaught=2, blackCaught=0,
                     blackBearing=False, whiteBearing=False, blackOutside=0, whiteOutside=0, ended=False)

def black_insertion() -> BackgammonState:
     return BackgammonState([0] * 23 , whiteCaught=0, blackCaught=1, blackBearing=False, whiteBearing=False, blackOutside=0, whiteOutside=0, ended = False)


"""

def test_unused_items():
     print(get_unused_items([4,4,4,4], [4,4]))
     print (get_unused_items([1,3], [3]))
"""

def test_generate_normal_black_1():
     game_state = get_starting_state()
     moves = generate_moves(game_state, is_black=True, dice=  [2])
     assert len(moves) == 4

def test_generate_normal_black_2():
     game_state = get_starting_state()
     moves = generate_moves(game_state, [2, 3])
     #print(moves)

def test_generate_normal_black_3():
     game_state = get_starting_state()
     moves = generate_moves(game_state, [2])
     #print(moves)
    # print(moves)

def test_one_before_bearing_move():
     game_state = one_step_before_bearing_off()
     moves = generate_moves(game_state, [3, 3])
     print(moves)


def test_white_insertion():
     game_state = white_insertion()
     moves = generate_moves(game_state=game_state, is_black=False, dice=[1,3])
     print(moves)

def test_black_insertion():
     game_state = black_insertion()
     moves = generate_moves(game_state=game_state, is_black=True, dice=[1, 1])
     print(moves)
     assert len(moves) == 1
     print("-----------------------------")



