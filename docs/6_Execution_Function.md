

# Run multiple models on the car with a custom execution function

The goal of this tutorial is to execute two models on the car, one following the track and one recognizing traffic signs. The car should behave accordingly.

## Prerequisites

Make sure you have two models prepared you want to test. For this tutorial, we use a model that follows a track as described [here](3_Autonomous_Driving.md) and another model which recognizes traffic signs as described [here](5_Customvision.md). Our second model uses a custom image preprocessor, we'll reuse this here. All in all, this is the code we'll start with:

```python
from autcar import Camera, Model
import numpy as np
import cv2

cam = Camera(rotation=-1)

class OwnModel(Model):
  def preprocess(self, image):
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    image = np.array([b,g,r]).transpose()
    h, w = image.shape[:2]
    min_dim = min(w,h)
    startx = w//2-(min_dim//2)
    starty = h//2-(min_dim//2)
    image = image[starty:starty+min_dim, startx:startx+min_dim]
    resized_image = np.array([cv2.resize(image, (224, 224), interpolation = cv2.INTER_LINEAR)])
    
    return resized_image
```
