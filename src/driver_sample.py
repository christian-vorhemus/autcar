from autcar import Car, Driver

car = Car()
driver = Driver("cntk_model.onnx", car, capture_interval=3, rotation=-1)
driver.start()