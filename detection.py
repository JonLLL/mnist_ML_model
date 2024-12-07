import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
import sys
from PIL import Image, ImageOps

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.VGG = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5,stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(16,32, kernel_size=5,stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
                             )
    self.fc1 = nn.Linear(32*7*7, 128)
    self.fc2 = nn.Linear(128,10)
    self.flat = nn.Flatten()

  def forward(self, x):
    x = self.VGG(x)
    x = self.flat(x)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# prediction function 
# takes the filepath of the image to predict and prints its prediction to console
def predict(filepath):
    image = read_image(filepath)/255.0
    model = MyModel()
    model.load_state_dict(torch.load('model_weights.pth', weights_only=True))

    with torch.no_grad():
       model.eval()
       y_pred = model(image.unsqueeze(0)).argmax(-1)

    print(y_pred.item())

# image transformation function 
def imagetransform(filepath):
    image_path = filepath
    file = image_path.split(".")
    image = Image.open(image_path)
    resized_image = image.resize((28, 28))
    inverted = ImageOps.invert(resized_image)
    grayscale_image = inverted.convert("L")
    new_name = "input_"+file[0]+".png"
    grayscale_image.save(new_name)

    return new_name

# to run files that are not transformed yet uncomment lines 59 and 60, this will call imagetransform function 
if __name__ == "__main__":
    # file_name = imagetransform(sys.argv[1])
    # predict(file_name)
    predict(sys.argv[1])

# For the input test images, i made them in ms paint, resize them to 28 by 28, inverted and then made them into grayscale.
# the expected arguments is an image path of an image with correct characteristics (i.e 28X28, inverted, grayscale)
# when running on hand drawn images with digits in the middle, it performed well (with some errors in classification for example inpur_nine4.png/ nine4.png(not transformed))
# when running on images where the digits where off to the side it also did not perform well either, with input_four_notmiddle.png it classified it as a 1 