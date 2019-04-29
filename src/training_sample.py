from autcar import Trainer

trainer = Trainer()
trainer.create_balanced_dataset(["src/ml/data/autcar_training_1", "src/ml/data/autcar_training_2"], outputfolder_path="src/ml/data/balanced_dataset")
trainer.train("src/ml/data/autcar_training_balanced_new", epochs=20)

trainer.test("driver_model.onnx", "src/ml/data/autcar_training_balanced_new/test_map.txt")