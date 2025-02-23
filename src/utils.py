
import random

def generate_dice_for_move() -> list[int]:
     first = random.randint(1, 6)
     second = random.randint(1, 6)
     if first == second:
          return [first, second] * 2
     else:
          return [first , second]