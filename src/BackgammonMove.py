from abc import ABC
from src.constants import LAST_INDEX_FIELD_INCLUDING_BEATON, FIRST_INDEX_FIELD_INCLUDING_BEATON

class BackgammonMove(ABC):

     def __init__(self, fromField : int , toField : int ,  insertionMove : bool = False, bearingOffMove : bool = False):
          #assert (toField >= FIRST_INDEX_FIELD_INCLUDING_BEATON and toField <= LAST_INDEX_FIELD_INCLUDING_BEATON) , "Index out of bounds for move"
          self.fromField : int = fromField
          self.toField : int  = toField
          self.insertionMove : bool = insertionMove
          self.bearingOffMove : bool = bearingOffMove
     

"""
     This class represents an Action A as describes in the Markov Decision Process for the White Player.

     White Stones are represented as negative Integers.

     If the white player wants to bear of a piece it should land on the field with index -1
"""

class BackgammenMoveWhite(BackgammonMove):

     def __init__(self, fromField, toField, insertionMove : bool = False , bearingOffMove : bool = False):
          assert fromField > toField , "White should move right to left"
          super().__init__(fromField, toField, insertionMove=insertionMove, bearingOffMove=bearingOffMove)
     
     def __str__(self):
          return f"({self.fromField} ,  {self.toField})"
     
     def __repr__(self):
          return f"({self.fromField} ,  {self.toField})"


"""
     This class represents an Action A as described in the Markov Decison Process for the Black Player.

     Black stones are represented as positive integer.

     If the black player wants to bear off the a piece it must land on the outside field with index 24
"""

class BackGammonMoveBlack(BackgammonMove):

     def __init__(self, fromField, toField, insertionMove : bool = False, bearingOffMove : bool = False):
          assert toField > fromField , "Black should move from left to right"
          super().__init__(fromField, toField, insertionMove=insertionMove, bearingOffMove=bearingOffMove)
     
     def __str__(self):
          return f"({self.fromField} ,  {self.toField})"
     
     def __repr__(self):
          return f"({self.fromField} ,  {self.toField})"