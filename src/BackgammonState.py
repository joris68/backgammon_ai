
"""
     This class Represents a State S in the Markov Decision Process
"""

class BackgammonState:

     def __init__(self, board , whiteCaught : int, blackCaught : int, blackBearing : bool , whiteBearing : bool):
          self.board : list[int] = board
          self.whiteCaught : int = whiteCaught
          self.blackCaught : int = blackCaught
          self.blackBearing : bool = blackBearing
          self.whiteBearing : bool = whiteBearing
     
     def __str__(self):
          return f"{self.board},  caughtWhite : {self.whiteCaught}, caughtBlack : {self.blackCaught}, whiteBearing : {self.whiteBearing} , blackbearing : {self.blackBearing}"
