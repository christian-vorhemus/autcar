from autcar import Car, Driver, Camera, Model

car = Car()
cam = Camera(rotation=-1)

model = Model("driver_keras.onnx")

driver = Driver(model, car, cam, execution_interval=2)
driver.start()