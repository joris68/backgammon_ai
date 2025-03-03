
"""
     This class Represents a State S in the Markov Decision Process.

     Black : positive Integers
     White : negative Integers

     The filed will be zero-indexed
"""


class BackgammonState:

     def __init__(self, board , whiteCaught : int, blackCaught : int, blackBearing : bool , whiteBearing : bool, ended : bool, blackOutside : int, whiteOutside : int):

          self.board : list[int] = board
          self.whiteCaught : int = whiteCaught
          self.blackCaught : int = blackCaught
          self.blackBearing : bool = blackBearing
          self.whiteBearing : bool = whiteBearing
          self.blackOutside : int = blackOutside
          self.whiteOutside : int = whiteOutside
          self.ended : bool = ended
     
     def __str__(self):
          return f"{self.board},  whiteCaught : {self.whiteCaught}, caughtBlack : {self.blackCaught}, whiteBearing : {self.whiteBearing} , blackbearing : {self.blackBearing} , blackOutside : {self.blackOutside},  whiteOutside : {self.whiteOutside} ended : {self.ended}"

     def __repr__(self):
          return f"{self.board},  whiteCaught : {self.whiteCaught}, caughtBlack : {self.blackCaught}, whiteBearing : {self.whiteBearing} , blackbearing : {self.blackBearing} , blackOutside : {self.blackOutside},  whiteOutside : {self.whiteOutside} ended : {self.ended}"
     
     def __eq__(self, other):
          if not isinstance(other, BackgammonState):
               return False
          return (self.board == other.board and
                    self.whiteCaught == other.whiteCaught and
                    self.blackCaught == other.blackCaught and
                    self.blackBearing == other.blackBearing and
                    self.whiteBearing == other.whiteBearing and
                    self.blackOutside == other.blackOutside and
                    self.whiteOutside == other.whiteOutside and
                    self.ended == other.ended)

     def __hash__(self):
        return hash((
            tuple(self.board),  
            self.whiteCaught,
            self.blackCaught,
            self.blackBearing,
            self.whiteBearing,
            self.blackOutside,
            self.whiteOutside,
            self.ended))