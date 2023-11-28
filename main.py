# to run type "python main.py" 

import os
import cv2
from random import randint
from deepface import DeepFace  
from pathlib import Path   
from aenum import MultiValueEnum

class Race(MultiValueEnum):
        African = "black"
        Asian = "asian"
        Caucasian = "white", "latino hispanic"
        Indian = "indian", "middle eastern"

current_race = Race.Asian

searchpath = f'./Equalizedface.tar/race_per_7000/{current_race.name}'

filepath = Path(searchpath)
files = [f.path for f in os.scandir(filepath) if f.is_dir()]

current_sample = 0

SAMPLE_SIZE = 100

file_iter = (i for i in files if current_sample < SAMPLE_SIZE)

correct = 0
incorrect = SAMPLE_SIZE

for i in file_iter:
        file_name = os.listdir(i)
        file_name = file_name[randint(0,len(file_name)-1)]
        name = i + "\\" + file_name
        name = name.split("\\")
        name = "/".join(name)
        name = "./" + name

        img = cv2.imread(name, cv2.IMREAD_COLOR)
        
        objs = DeepFace.analyze(img, 
                actions = ['race'],
                enforce_detection = False,
        )

        result = Race(objs[0]["dominant_race"])

        if result == current_race:
                correct += 1
                incorrect -= 1

        current_sample += 1

print("-------------------------RESULTS--------------------------")
print(f'Race Tested: {current_race.name}')
print("----------------------------------------------------------")
print(f'Total Pictures Tested {SAMPLE_SIZE}')
print("----------------------------------------------------------")
print(f'Total Correctly Identified {correct}')
print(f'Percent Correct: {round((correct/SAMPLE_SIZE) * 100, 1)}%')
print("----------------------------------------------------------")
print(f'Total Incorrectly Identified {incorrect}')
print(f'Percent Incorrect: {round((incorrect/SAMPLE_SIZE) * 100, 1)}%')