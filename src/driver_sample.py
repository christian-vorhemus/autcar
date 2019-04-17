from autcar import Car, Driver

car = Car()
driver = Driver("rh.onnx", car, capture_interval=4)
driver.start()