
from src.BackgammonState import BackgammonState
import logging

logger = logging.getLogger(__name__)


ERRORS = {
     "BLACK_STONES" : "Black stones aint adding up",
     "WHITE_STONES" : "White stones aint adding up",
     "OUTSIDE_AND_BEARING" : "If there are stones outside Bearing should be false"
}

def _black_number_stones(game_state : BackgammonState) -> bool:
     stones = 0
     for _ , x in enumerate(game_state.board):
          if x > 0:
               stones += x
     
     return (game_state.blackCaught + stones + game_state.blackOutside) == 15


def _white_number_stones(game_state : BackgammonState) -> bool:
     stones = 0
     for _ , x in enumerate(game_state.board):
          if x < 0:
               stones += abs(x)
     return (game_state.whiteCaught + stones + game_state.whiteOutside) == 15

def _outside_and_bearing(game_state : BackgammonState) -> bool:
     if game_state.blackCaught > 0 and game_state.blackBearing:
         return False
     
     if game_state.whiteCaught > 0 and game_state.whiteBearing:
          return False
     
     return True

def backgammonstate_invariant(game_state_before : BackgammonState , game_state_after : BackgammonState):
     if not _black_number_stones(game_state=game_state_after):
          logger.info(f"Game state before: {game_state_before} Invalid Game State: {game_state_after}")
          raise Exception(ERRORS["BLACK_STONES"])
     if not _white_number_stones(game_state_after):
          logger.info(f"Game state before: {game_state_before} Invalid Game State: {game_state_after}")
          raise Exception(ERRORS["WHITE_STONES"])
     #if not _invariant_stones_outside(game_state=game_state):
      #    raise Exception(ERRORS["STONES_OUTSIDE"])
     if not _outside_and_bearing(game_state=game_state_after):
          logger.info(f"Game state before: {game_state_before} Invalid Game State: {game_state_after}")

