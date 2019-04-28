from autcar import Car, Driver

car = Car()
driver = Driver("driver_model.onnx", car, capture_interval=2, rotation=-1)
driver.start()