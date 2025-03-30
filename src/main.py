from src.BackgammonModel import BackgammonModel
from src.Game import GammonMonteCarlo
from pathlib import Path
import statistics
import json

def save_result(result : dict, name : str) -> None:
     with open(Path(f"src/results/{name}"), "w") as file:
          json.dump(result, file)

def calc_result_statistics(res : dict) -> dict:
     return {
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

def main():

     model_0 = BackgammonModel(0.8, 0.1, 0, model_path=Path("src/models/0_g_training.pt"))
     model_0.train_model()

     monte_0 = GammonMonteCarlo(1000, model_path="src/models/50000_g_training.pt")
     res_0 = monte_0.test_value_function()
     result_0 =  calc_result_statistics(res=res_0)
     save_result(result=result_5000, name="result_0")

     ######################################################

     model_5000 = BackgammonModel(0.8, 0.1, 5000, model_path=Path("src/models/5000_g_training.pt"))
     model_5000.train_model()

     monte_5000 = GammonMonteCarlo(1000, model_path="src/models/50000_g_training.pt")
     res_5000 = monte_5000.test_value_function()
     result_5000 =  calc_result_statistics(res=res_5000)
     save_result(result=result_5000, name="result_5000")

     
