from src.BackgammonEngine import generate_moves
from src.BackgammonState import BackgammonState
import numpy as np
import logging
import matplotlib.pyplot as plt
from src.BackgammonModel import BackgammonModel , load_model
from src.utils import generate_dice_for_move
from src.constants import STARTING_GAME_STATE
import statistics
import json


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def is_beat_move_black(curr : BackgammonState, next : BackgammonState) -> int:
     if next.whiteCaught > curr.whiteCaught:
          return 1
     else:
          return 0

def is_beat_move_white(curr : BackgammonState, next : BackgammonState) -> int:
     if next.blackCaught > curr.blackCaught:
          return 1
     else:
          return 0

def get_open_positions_black(curr: BackgammonState) -> int:
     counter = 0
     for x in curr.board:
          if x == 1:
               counter += 1
     return counter


def get_open_positions_white(curr : BackgammonState) -> int:
     counter = 0
     for x in curr.board:
          if x == -1:
               counter +=1
     return counter


class GammonMonteCarlo:
     
     def __init__(self, number_of_games : int, model_path : str):
          self.number_of_games : int = number_of_games
          self.valueFunction : BackgammonModel = load_model(model_path)

     def test_value_function(self) -> dict:
          AI_won = 0
          uniform_won = 0
          beat_moves_model = []
          beat_moves_uniform = []
          open_positions_model = []
          open_positions_uniform = []
          game_lengths = []

          for x in range(self.number_of_games):
               if x % 10 == 0:
                    logger.info(f"starting game {x}")
               is_black = np.random.rand() > 0.5
               curr_game_state = STARTING_GAME_STATE

               beat_moves_model_game = 0
               beat_moves_uniform_game = 0
               open_positions_uniform_game = 0
               open_positions_model_game = 0
               play_counter = 0

               while not curr_game_state.ended:
                    the_dice = generate_dice_for_move()
                    if is_black:
                         next_state = self.valueFunction.infer_state(game_state=curr_game_state, dice=the_dice, is_black=is_black)
                         beat_moves_model_game += is_beat_move_black(curr=curr_game_state, next=next_state)
                         open_positions_model_game += get_open_positions_black(curr=next_state)
                         curr_game_state = next_state
                    else:
                         poss_next_states = generate_moves(game_state=curr_game_state, is_black=is_black, dice=the_dice)
                         next_state = np.random.choice(poss_next_states)
                         beat_moves_uniform_game += is_beat_move_white(curr=curr_game_state, next=next_state)
                         open_positions_uniform_game += get_open_positions_white(curr=next_state)
                         curr_game_state = next_state
                    
                    is_black = not is_black
                    play_counter += 1
               
               beat_moves_model.append(beat_moves_model_game)
               beat_moves_uniform.append(beat_moves_uniform_game)
               game_lengths.append(play_counter)
               open_positions_model.append(open_positions_model_game)
               open_positions_uniform.append(open_positions_uniform_game)

               if curr_game_state.whiteOutside == 15:
                    uniform_won += 1
               else:
                    AI_won += 1
          

          return {
               "model_won" : AI_won,
               "uniform_won" : uniform_won,
               "beat_moves_model" : beat_moves_model,
               "beat_moves_uniform" : beat_moves_uniform,
               "open_positions_model": open_positions_model,
               "open_positions_uniform" : open_positions_uniform,
               "game_lengths" : game_lengths
          }
                    

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




