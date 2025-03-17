from src.BackgammonState import BackgammonState
"""
     To represent stones that have been beaton by other player the:

     24 : (fromField) virtual field to reresent stones coming in for the black player.
     -1 : (fromField) virtual field to represent stones coming in for the white player.
     
"""
LAST_INDEX_FIELD_INCLUDING_BEATON = 24
FIRST_INDEX_FIELD_INCLUDING_BEATON = -1

LAST_INDEX_FIELD_BOARD = 23
FIRST_INDEX_FIELD_BOARD = 0


STARTING_GAME_STATE = BackgammonState([2, 0, 0, 0, 0, -5,   0, -3, 0 ,0 ,0 , 5,   -5, 0,0,0,3,0,   5, 0,0,0,0,-2], whiteCaught=0, blackCaught=0,
                     blackBearing=False, whiteBearing=False, blackOutside=0, whiteOutside=0, ended=False)