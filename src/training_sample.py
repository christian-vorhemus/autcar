from autcar import Trainer

trainer = Trainer()
trainer.create_balanced_dataset("src/ml/data/autcar_training_1", outputfolder_path="src/ml/data/teee")
trainer.train("src/ml/data/autcar_training_balanced_new", epochs=20)

# accuracy, f1-score, confusion matrix
trainer.test("car_cntk.model")