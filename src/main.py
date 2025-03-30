from src.BackgammonModel import BackgammonModel
from src.Game import GammonMonteCarlo
from pathlib import Path
import statistics
import json
import logging

logger = logging.getLogger(__name__)

def save_result(result : list[dict], name : str) -> None:
     with open(Path(f"src/results/{name}"), "w") as file:
          json.dump(result, file)

def calc_result_statistics(res : dict) -> dict:
     a = {
          "model" : res["model_won"],
          "uniform" : res["uniform_won"],
          "avg_beat_ai" : statistics.mean(res["beat_moves_model"]),
          "std_beat_ai" : statistics.stdev(res["beat_moves_model"]),
          "avg_beat_uniform" : statistics.mean(res["beat_moves_uniform"]),
          "std_beat_uniform" : statistics.stdev(res["beat_moves_uniform"]),
          "avg_open_positions_model": statistics.mean(res["open_positions_model"]),
          "std_open_positions_model": statistics.stdev(res["open_positions_model"]),
          "avg_open_positions_uniform": statistics.mean(res["open_positions_uniform"]),
          "std_open_positions_uniform": statistics.stdev(res["open_positions_uniform"]),
          "avg_game_length": statistics.mean(res["game_lengths"]),
          "std_game_length":statistics.stdev(res["game_lengths"])
     }
     return a

def main():

     MONTE_CARLO_GAMES = 1000

     model_0 = BackgammonModel(0.8, 0.1, 0, model_path=Path("src/models/0_g_training.pt"))
     model_0.train_model()

     monte_0 = GammonMonteCarlo(MONTE_CARLO_GAMES, model_path=Path("src/models/0_g_training.pt"))
     res_0 = monte_0.test_value_function()
     result_0 =  calc_result_statistics(res=res_0)
     save_result(result=result_0, name="result_0.json")

     ######################################################

     model_5000 = BackgammonModel(0.8, 0.1, 5000, model_path=Path("src/models/5000_g_training.pt"))
     model_5000.train_model()

     monte_5000 = GammonMonteCarlo(MONTE_CARLO_GAMES, model_path=Path("src/models/5000_g_training.pt"))
     res_5000 = monte_5000.test_value_function()
     result_5000 = calc_result_statistics(res=res_5000)
     save_result(result=result_5000, name="result_5000.json")

     ###############################################################

     model_50000 = BackgammonModel(0.8, 0.1, 50000, model_path=Path("src/models/50000_g_training.pt"))
     model_50000.train_model()

     monte_50000 = GammonMonteCarlo(MONTE_CARLO_GAMES, model_path=Path("src/models/50000_g_training.pt"))
     res_50000 = monte_50000.test_value_function()
     result_50000 = calc_result_statistics(res=res_50000)
     save_result(result=result_50000, name="result_50000.json")

def test_models_repeatedly(runs : int, models : list[str], names : list[str]):

     for m in range(len(models)):
          logger.info(f"starting to eval model : {m}")
          result_list = []

          for x in range(runs):
               logger.info(f"starting run : {x}")
               monte = GammonMonteCarlo(1000, model_path=Path(models[m]))
               res = monte.test_value_function()
               result = calc_result_statistics(res=res)
               result_list.append(result)
     
          save_result(result=result_list, name=names[m])


#test_models_repeatedly(5, ["src/models/5000_g_training.pt", "src/models/50000_g_training.pt"], ["repeat_5000.json", "repeat_50000.json"])

def train_model(games : int):
     logger.info(f"starting to train model with games : {games}")
     model = BackgammonModel(0.8, 0.1, games, model_path=Path(f"src/models/{games}_g_training.pt"))
     model.train_model()

train_model(games=100000)
train_model(games=150000)



     
