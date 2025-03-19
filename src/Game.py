from src.BackgammonEngine import generate_moves
from src.BackgammonState import BackgammonState
import numpy as np
import logging
import matplotlib.pyplot as plt
from src.BackgammonModel import BackgammonModel , load_model
from src.utils import generate_dice_for_move
from src.constants import STARTING_GAME_STATE


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
          self.valueFunction : BackgammonModel = load_model()

     def test_value_function(self):
          AI_won = 0
          uniform_won = 0

          for x in range(self.number_of_games):
               if x % 10 == 0:
                    logger.info(f"starting game {x}")
               is_black = np.random.rand() > 0.5
               curr_game_state = STARTING_GAME_STATE

               while not curr_game_state.ended:
                    the_dice = generate_dice_for_move()
                    if is_black:
                         curr_game_state = self.valueFunction.infer_state(game_state=curr_game_state, dice=the_dice, is_black=is_black)
                    else:
                         poss_next_states = generate_moves(game_state=curr_game_state, is_black=is_black, dice=the_dice)
                         curr_game_state = np.random.choice(poss_next_states)
                    
                    is_black = not is_black
               
               if curr_game_state.whiteOutside == 15:
                    uniform_won += 1
               else:
                    AI_won += 1
          
          return AI_won, uniform_won
                    




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
          counter = 0
          while not curr_game_state.ended:
               
               the_dice = generate_dice_for_move()
               poss_next_states = generate_moves(curr_game_state, is_black= is_black, dice=the_dice)
               moves_counter.append(len(poss_next_states))
               curr_game_state = np.random.choice(poss_next_states)
               is_black = not is_black
               counter += 1
          
          won = "black" if curr_game_state.blackOutside == 15 else "white"

          return counter, won, moves_counter
     
     def beat_move_executed(self , curr : BackgammonState, next: BackgammonState, is_black : bool) -> bool:
          if is_black:
              return  curr.whiteCaught < next.whiteCaught
          else:
               return curr.blackCaught < next.blackCaught
     
     def count_open_positions(self, game_state : BackgammonState, is_black : bool) -> int:
          open_positions = 0
          if is_black:
               for x in game_state.board:
                    if x == 1:
                         open_positions += 1
          else:
               for x in game_state.board:
                    if x == -1:
                         open_positions += 1
          
          return open_positions



     # black beat moves, black open positions, white beat moves, white open positions
     def _play_game_save(self) -> tuple[int, int, int, int]:
          is_black = True if np.random.rand() > 0.5 else False
          curr_game_state = STARTING_GAME_STATE

          black_beat_moves = 0
          white_beat_moves = 0
          white_open_positions = 0
          black_open_positions = 0

          while not curr_game_state.ended:
               the_dice = generate_dice_for_move()
               poss_next_states = generate_moves(curr_game_state, is_black= is_black, dice=the_dice)
               next_state = np.random.choice(poss_next_states)
               if self.beat_move_executed(curr=curr_game_state, next=next_state , is_black=is_black):
                    if is_black:
                         black_beat_moves += 1
                    else:
                         white_beat_moves += 1
               curr_game_state = next_state
               open_pos = self.count_open_positions(game_state=curr_game_state, is_black=is_black)
               if is_black:
                    black_open_positions += open_pos
               else:
                    white_open_positions += open_pos
               is_black = not is_black
          
          return black_beat_moves, black_open_positions, white_beat_moves, white_open_positions

     
     def _simulate_save_games(self) -> tuple[list[int], list[int], list[int], list[int]]:
          beat_moves_white_per_game = []
          beat_moves_black_per_game = []
          open_positions_black_per_game = []
          open_positions_white_per_game = []

          for x in self.number_of_games:
               black_beat_moves, black_open_positions, white_beat_moves, white_open_positions = self._play_game_save()
               beat_moves_black_per_game.append(black_beat_moves)
               beat_moves_white_per_game.append(white_beat_moves)
               open_positions_black_per_game.append(black_open_positions)
               open_positions_white_per_game.append(white_open_positions)
          


def plot_average_game_length(game_lengths : list[int]) -> None:
     plt.hist(game_lengths, bins=10, orientation='vertical', color='purple', alpha=0.7)
     plt.xlabel('Plays per Game')
     plt.ylabel('Frequency')
     plt.title('Distribution of Plays per Game')
     plt.savefig('distribution_plays_per_game.png', dpi=300)
     plt.show()


if __name__ == '__main__':
     monte = GammonMonteCarlo(5)
     #plays_per_game , won, moves_counter = monte._simulate_games()
     #plot_average_game_length(game_lengths=plays_per_game)
     AI, uniform = monte.test_value_function()

     logger.info(f"AI : {AI}")
     logger.info(f"Uniform : {uniform}")




