
from pathlib import Path
import json
import statistics

FILES = ["repeat_5000.json", "repeat_50000.json", "repeat_100000.json", "repeat_150000.json", "repeat_200000.json", "repeat_250000.json"]

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