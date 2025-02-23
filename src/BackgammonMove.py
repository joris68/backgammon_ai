from abc import ABC
from constants import LAST_INDEX_FIELD, FIRST_INDEX_FIELD

class BackgammonMove(ABC):

     def __init__(self, fromField : int , toField : int ):
          assert (toField >= FIRST_INDEX_FIELD and toField <= LAST_INDEX_FIELD) , "Index out of bounds for move"
          self.fromField : int = fromField
          self.toField : int  = toField

"""
     This class represents an Action A as describes in the Markov Decision Process for the White Player
"""

class BackgammenMoveWhite(BackgammonMove):

     def __init__(self, fromField, toField):
          assert fromField < toField , "White should move left to right"
          super().__init__(fromField, toField)


"""
     This class represents an Action A as described in the Markov Decison Process for the Black Player
"""

class BackGammonMoveBlack(BackgammonMove):

     def __init__(self, fromField, toField):
          assert toField < fromField , "Black should move from right to left"
          super().__init__(fromField, toField)