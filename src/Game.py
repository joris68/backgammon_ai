from src.BackgammonEngine import generate_moves
from src.BackgammonState import BackgammonState
from src.utils import generate_dice_for_move
import numpy as np
import logging


STARTING_GAME_STATE = BackgammonState([2, 0, 0, 0, 0, -5,   0, -3, 0 ,0 ,0 , 5,   -5, 0,0,0,3,0,   5, 0,0,0,0,-2], whiteCaught=0, blackCaught=0,
                     blackBearing=False, whiteBearing=False, blackOutside=0, whiteOutside=0, ended=False)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def play_game():

     is_black = True
     curr_game_state = STARTING_GAME_STATE
     logger.info(f"starting game state: {curr_game_state}")
     counter = 0
     while not curr_game_state.ended:
          logger.info(f"the {counter}th move")
          the_dice = generate_dice_for_move()
          logger.info(f"Dice : {the_dice}")
          poss_next_states = generate_moves(curr_game_state, is_black= is_black, dice=the_dice)
          logger.info(f"got {len(poss_next_states)} moves")
          curr_game_state = np.random.choice(poss_next_states)
          is_black = not is_black
          counter += 1
          logger.info(f"game state after move {counter} : {curr_game_state}")

play_game()
          