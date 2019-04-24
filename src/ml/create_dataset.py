import glob
import os
import csv
import ast
from shutil import copyfile

if not os.path.exists("src/ml/data/merged_new"):
    os.makedirs("src/ml/data/merged_new")

files = os.listdir("src/ml/data")

num = 0
for file in files:
    if(file == "merged" or file == "autcar_training_balanced" or file == "merged_new"):
        continue
    with open("src/ml/data/"+file+"/training.csv") as training_file:
        csv_reader = csv.reader(training_file, delimiter=';')
        print(file)

        for row in csv_reader:
            filename = row[0]
            try:
                command = ast.literal_eval(row[1])
            except:
                continue
            cmd_type = command["type"]

            if(cmd_type == "stop"):
                continue

            copyfile("src/ml/data/"+file+"/"+filename, "src/ml/data/merged_new/snapshot_"+str(num)+".png")

            text = "snapshot_"+str(num)+".png;" + cmd_type + "\n"
            nf = open('src/ml/data/merged_new/training.csv', 'a', encoding="utf-8")
            nf.write(text)
            nf.close()

            num = num + 1
