from autcar import Car, Driver

car = Car()
driver = Driver("rh.onnx", car, capture_interval=3)
driver.start()