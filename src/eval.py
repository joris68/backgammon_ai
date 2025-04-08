
from pathlib import Path
import json
import statistics

FILES = ["20000_500_high_reward_positive.json", "40000_500_high_reward_positive.json", "60000_500_high_reward_positive.json", "80000_500_high_reward_positive.json", "100000_500_high_reward_positive.json"]

PATH = "src/results/"

def average_json_array(files : list[str]) -> None:
     results = {}

     for file_name in files:

          with open(Path(PATH + file_name), "r") as file:
              data = json.load(file)
          keys = data[0].keys()
          averaged_dict = {key: statistics.mean(d[key] for d in data) for key in keys}

          print(f"for the file : {file_name} : ")
          print(averaged_dict)
          print("--------------------------")

average_json_array(files=FILES)