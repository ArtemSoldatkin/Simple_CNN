import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

PATH_TO_TRAIN_DATASET = "Dataset/Train"
PATH_TO_MODEL = "model.model"
BATCH_SIZE = 5
EPOCHS = 6
CLASSES = ('''classes''')
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32*53*53, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 32 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x# create a complete CNN



def load_model():
    model = CNNModel()
    model.load_state_dict(torch.load(PATH_TO_MODEL, map_location = device))
    return model

model = load_model()
model.to(device)
'''
model = CNNModel()
model.to(device)
'''
def save_model():
    torch.save(model.state_dict(), PATH_TO_MODEL)

def prepare_to_fit():
    transformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225])
    ])
    train_set = datasets.ImageFolder(PATH_TO_TRAIN_DATASET,
        transform = transformations)
    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size = BATCH_SIZE, shuffle = True)
    return train_loader

def fit():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    train_loader = prepare_to_fit()
    model.train()
    for epoch in range(EPOCHS):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(inputs)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    save_model()


def process_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    width, height = img.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    img = np.array(img)
    img = img.transpose((2, 0, 1))
    img = img/255
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    img = img[np.newaxis,:]
    image = torch.from_numpy(img)
    image = image.float()
    return image

def predict(image):
    model.eval()
    image.to(device)
    model.to("cpu")
    output = model.forward(image)
    output = torch.exp(output)
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()

#fit()

image = process_image('''path to image''')

top_prob, top_class = predict(image)
top_class = CLASSES[top_class]
print("The model is ", top_prob*100, "% certain that the image has a predicted class of ", top_class  )
