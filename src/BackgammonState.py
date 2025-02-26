
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
          # if this is set to true black can start removing pieces
          self.blackBearing : bool = blackBearing
          self.whiteBearing : bool = whiteBearing
          self.blackOutside : int = blackOutside
          self.whiteOutside : int = whiteOutside
          self.ended : bool = ended
     
     def __str__(self):
          return f"{self.board},  whiteCaught : {self.whiteCaught}, caughtBlack : {self.blackCaught}, whiteBearing : {self.whiteBearing} , blackbearing : {self.blackBearing}"

     def __repr__(self):
          return f"{self.board},  whiteCaught : {self.whiteCaught}, caughtBlack : {self.blackCaught}, whiteBearing : {self.whiteBearing} , blackbearing : {self.blackBearing} , blackOutside : {self.blackOutside},  whiteOutside : {self.whiteOutside}"