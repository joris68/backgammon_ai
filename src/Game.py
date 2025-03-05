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


class GammonMonteCarlo:
     
     def __init__(self, number_of_games : int):
          self.number_of_games : int = number_of_games



     def _simulate_games(self) -> tuple[list[int], list[str], list[list[int]]]:
          turns_summary : list[int] = []
          winners  : list[str] = []
          number_moves : list[list[int]] = []
          successful = 0
          while successful < self.number_of_games: 
               try:
                    turns , won, moves = self._play_game()
                    turns_summary.append(turns)
                    winners.append(won)
                    number_moves.append(moves)
                    successful += 1
               except Exception as e:
                    logger.error(e)

          return turns_summary, winners, number_moves

     def _play_game(self) -> tuple[int , str, list[int]]:
          """
          Gives back the number of turns, who has won, a list[int] the number of moves per turn 
          """

          is_black = True if np.random.rand() > 0.5 else False
          curr_game_state = STARTING_GAME_STATE
          moves_counter = []

          #logger.info(f"starting game state: {curr_game_state}")
          counter = 0
          while not curr_game_state.ended:
               #logger.info(f"the {counter}th move")
               #logger.info(f"Game state before {curr_game_state}")
               the_dice = generate_dice_for_move()
               #logger.info(f"Dice : {the_dice}")
               #logger.info(f"Turn {"black" if is_black else "white"}")
               poss_next_states = generate_moves(curr_game_state, is_black= is_black, dice=the_dice)
               moves_counter.append(len(poss_next_states))
               #logger.info(f"got {len(poss_next_states)} moves")
               curr_game_state = np.random.choice(poss_next_states)
               is_black = not is_black
               counter += 1
               #logger.info(f"game state after move {counter} : {curr_game_state}")
          
          won = "black" if curr_game_state.blackOutside == 15 else "white"

          return counter, won, moves_counter


if __name__ == '__main__':
     monte = GammonMonteCarlo(20)
     print(monte._simulate_games())
