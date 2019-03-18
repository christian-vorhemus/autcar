import glob
import os
import csv
import ast
from shutil import copyfile

if not os.path.exists("src/mlscripts/data/merged"):
    os.makedirs("src/mlscripts/data/merged")

files = os.listdir("src/mlscripts/data")

num = 0
for file in files:
    with open("src/mlscripts/data/"+file+"/training.csv") as training_file:
        csv_reader = csv.reader(training_file, delimiter=';')
        print(file)
        if(file == "merged"):
            continue

        for row in csv_reader:
            filename = row[0]
            try:
                command = ast.literal_eval(row[1])
            except:
                continue
            cmd_type = command["type"]

            if(cmd_type == "stop"):
                continue

            copyfile("src/mlscripts/data/"+file+"/"+filename, "src/mlscripts/data/merged/snapshot_"+str(num)+".png")

            text = "snapshot_"+str(num)+".png;" + cmd_type + "\n"
            nf = open('src/mlscripts/data/merged/training.csv', 'a', encoding="utf-8")
            nf.write(text)
            nf.close()

            num = num + 1
