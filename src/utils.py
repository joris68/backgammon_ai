from src.BackgammonState import BackgammonState
import torch
import random
import logging

logger = logging.getLogger(__name__)

def generate_dice_for_move() -> list[int]:
     first = random.randint(1, 6)
     second = random.randint(1, 6)
     if first == second:
          return [first, second] * 2
     else:
          return [first , second]


def encode_field(stones : int, store : list[float]) -> None:
     if stones == 0:
          store.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ])
          return
     
     # black
     if stones == 1:
          store.extend([1.0, 0.0, 0.0, 0.0])
          store.extend([0.0, 0.0, 0.0, 0.0])
          return
     
     if stones == 2:
          store.extend([1.0 ,1.0 , 0.0, 0.0])
          store.extend([0.0, 0.0, 0.0, 0.0])
          return
     
     if stones == 3:
          store.extend([1.0, 1.0 , 1.0 , 0.0])
          store.extend([0.0, 0.0, 0.0, 0.0])
          return
     
     if stones > 3:
          a = (stones -3) / 2
          store.extend([1.0, 1.0, 1.0, a])
          store.extend([0.0, 0.0, 0.0, 0.0])
          return
     # white
     if stones == -1:
          store.extend([0.0, 0.0, 0.0, 0.0])
          store.extend([1.0, 0.0, 0.0, 0.0])
          return
     
     if stones == -2:
          store.extend([0.0, 0.0, 0.0, 0.0])
          store.extend([1.0 ,1.0 , 0.0, 0.0])
          return
     
     if stones == -3:
          store.extend([0.0, 0.0, 0.0, 0.0])
          store.extend([1.0, 1.0 , 1.0 , 0.0])
          return
     
     if stones < -3:
          store.extend([0.0, 0.0, 0.0, 0.0])
          a = (abs(stones) -3) / 2
          store.extend([1.0, 1.0, 1.0, a])
          return


def encode_outside(outside : int) -> float:
     return outside / 2

def encode_borne_off(bourne_off : int) -> float:
     return bourne_off / 15

def encode_turn(is_black : bool) -> list[float]:
     if is_black:
          return [1.0 , 0.0]
     else:
          return [0.0, 1.0]



def encode_backgammonstate(game_state : BackgammonState, is_black : bool) -> torch.Tensor:

     list_tensor = []
     field_encoding = []
     for field in game_state.board:
          encode_field(field, store=field_encoding)
     
     list_tensor.extend(field_encoding)
     list_tensor.append(encode_outside(game_state.whiteCaught))
     list_tensor.append(encode_outside(game_state.blackCaught))
     list_tensor.append(encode_borne_off(game_state.whiteOutside))
     list_tensor.append(encode_borne_off(game_state.blackOutside))
     list_tensor.extend(encode_turn(is_black=is_black))

     return torch.tensor(list_tensor, dtype=float)
