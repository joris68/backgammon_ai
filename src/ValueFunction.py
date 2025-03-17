import torch
import torch.nn as nn
from src.BackgammonEngine import generate_moves
from src.constants import STARTING_GAME_STATE
from src.BackgammonState import BackgammonState
from src.utils import generate_dice_for_move, encode_backgammonstate
import torch.optim as optim
import numpy as np
import logging
import copy


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)



class ValueFunction(nn.Module):

     def __init__(self , lambda_parameter : float, learning_rate : float, training_games : int):
          super(ValueFunction, self).__init__()
          self.lambda_parameter = lambda_parameter
          self.learning_rate = learning_rate
          self.training_games = training_games
          self.network = self._init_mlp()

     
     def _init_mlp(self) -> nn.ModuleList:
          layers = nn.ModuleList()
          first_layer = nn.Linear(in_features=198 ,out_features=80, dtype=float)
          second_layer = nn.Linear(in_features=80 , out_features=2,  dtype=float)
          nn.init.normal_(first_layer.weight, mean=0.0, std=1.0)
          nn.init.normal_(second_layer.weight, mean=0.0, std=1.0)
          layers.append(first_layer)
          layers.append(nn.Sigmoid())
          layers.append(second_layer)
          layers.append(nn.Sigmoid())
          return layers
     
     def forward(self, input_game_state : torch.Tensor, no_grad : bool) -> torch.Tensor:
          if no_grad:
               with torch.no_grad():
                    for layer in self.network:
                         input_game_state = layer(input_game_state)
                    
                    return input_game_state
          else:
               for layer in self.network:
                    input_game_state = layer(input_game_state)
                    
               return input_game_state

     def get_highest_prob_index_white(self, poss_next_state: list[BackgammonState], is_blacks_turn: bool) -> int:

          predictions = []
          for state in poss_next_state:
                    out = self.forward(encode_backgammonstate(game_state=state, is_black=is_blacks_turn), no_grad=True)
                    predictions.append(out)

          outputs = torch.stack(predictions)
          max_index = torch.argmax(outputs[:, 0]).item()

          return max_index
          

     def get_highest_prob_index_black(self, poss_next_state : list[BackgammonState], is_blacks_turn : bool) -> int:

          predictions = []
          for state in poss_next_state:
                    out = self.forward(encode_backgammonstate(game_state=state, is_black=is_blacks_turn), no_grad=True)
                    predictions.append(out)

          outputs = torch.stack(predictions)
          logger.info(outputs) 
          max_index = torch.argmax(outputs[:, 1]).item()
          logger.info(max_index)

          return max_index



     def TD_Error(self, reward_next : torch.Tensor, eval_next : torch.Tensor, eval_prev : torch.Tensor | list[float]) -> float:
          if isinstance(eval_prev, torch.Tensor):
               return reward_next + eval_next - eval_prev
          else:
               return reward_next + eval_next - torch.tensor(eval_prev)

     def get_reward_vector(self, game_state : BackgammonState) -> torch.Tensor:

          if game_state.whiteOutside == 15:
               return torch.tensor([1.0, 0.0])
          if game_state.blackOutside == 15:
               return torch.tensor([0.0, 1.0])
          
          return torch.tensor([0.0, 0.0])


     def train_model(self) -> None:
          
          optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)

          for x in range(self.training_games):
               logger.info(f"starting with game : {x}")
               e_t = torch.tensor([0.0, 0.0], dtype=float, requires_grad=False)
               is_blacks_turn = np.random.rand() > 0.5
               curr_game_state = STARTING_GAME_STATE
               
               play_counter = 0
               eval_curr = self.forward(encode_backgammonstate(curr_game_state, is_black=is_blacks_turn), no_grad=True)
               while not curr_game_state.ended:
                    optimizer.zero_grad()
                    poss_next_state = generate_moves(game_state=curr_game_state, is_black=is_blacks_turn, dice=generate_dice_for_move())
                    index_next_state = None
                    if is_blacks_turn:
                         index_next_state = self.get_highest_prob_index_black(poss_next_state=poss_next_state, is_blacks_turn=is_blacks_turn)
                    else:
                         index_next_state = self.get_highest_prob_index_white(poss_next_state=poss_next_state, is_blacks_turn=is_blacks_turn)
                    
                    eval_next =  self.forward(input_game_state=encode_backgammonstate(poss_next_state[index_next_state], is_black=is_blacks_turn), no_grad=False)
                    reward_next = self.get_reward_vector(poss_next_state[index_next_state])
                    td_error = self.TD_Error(reward_next=reward_next, eval_next=eval_next, eval_prev=eval_curr)
                    if isinstance(eval_curr, torch.Tensor):
                         e_t = self.lambda_parameter * e_t + eval_curr
                    else:
                         e_t = self.lambda_parameter * e_t + torch.tensor(eval_curr)
                    complete_error = td_error * e_t
                   
                    grad_output = torch.ones_like(complete_error)
                    complete_error.backward(grad_output)
                    optimizer.step()
                    curr_game_state = poss_next_state[index_next_state]
                    is_blacks_turn = not is_blacks_turn
                    eval_curr = eval_next.tolist()
                    play_counter +=1
               
               logger.info(f"played : {play_counter} games in game : {x} ")

if __name__=='__main__':
     value_function = ValueFunction(0.8, 0.1, 1)
     value_function.train_model()




                    



                   







          
     






