import torch
from torch.nn import nn
from src.BackgammonEngine import generate_moves
from src.constants import STARTING_GAME_STATE
from src.BackgammonState import BackgammonState
import numpy as np


class ValueFunction(nn.Module):

     def __init__(self , lambda_parameter : float):
          super(ValueFunction, self).__init__()
          self.lambda_parameter = lambda_parameter
          self.network = self._init_mlp()

     
     def _init_mlp(self) -> nn.ModuleList:
          layers = nn.ModuleList()
          first_layer = nn.Linear(in_features=198 ,out_features=80)
          second_layer = nn.Linear(in_features=80 , out_features=1)
          nn.init.normal_(first_layer.weight, mean=0.0, std=1.0)
          nn.init.normal_(second_layer.weight, mean=0.0, std=1.0)
          layers.append(first_layer)
          layers.append(nn.Sigmoid())
          layers.append(second_layer)
          layers.appedn(nn.Sigmoid())
          return layers
     
     def forward(self, input_game_state : torch.Tensor) -> torch.Tensor:
          pass

     def get_highest_prob_index(poss_next_state : list[BackgammonState]) -> tuple[int, torch.Tensor]:

          predictions : list[torch.Tensor] = []

          for state in poss_next_state:
               pass


     def train_model(self) -> None:
          


          for x in range(100):
               delta = 0
               e_t = 0
               is_blacks_turn = True if np.random.rand() > 0.5 else False
               curr_game_state = STARTING_GAME_STATE
               while not curr_game_state.ended:
                    eval_curr = self.forward()
                    poss_next_state = generate_moves(game_state=curr_game_state, is_black=is_blacks_turn)
                    



                   







          
     






