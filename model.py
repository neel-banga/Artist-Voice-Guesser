import os
import scipy.io.wavfile as wavfile
import torch
import torch.nn as nn
import random

# Create the network

class Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, hidden_size)
        self.fc9 = nn.Linear(hidden_size, hidden_size)
        self.fc10 = nn.Linear(hidden_size, hidden_size)
        self.fc11 = nn.Linear(hidden_size, hidden_size)
        self.fc12 = nn.Linear(hidden_size, hidden_size)
        self.fc13 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size) 

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        x = self.fc10(x)
        x = self.fc11(x)
        x = self.fc12(x)
        x = self.fc13(x)
        x = self.output(x)

        return x


# Here we convert our audio file to a tensor; we do this so we can pass the audio tensors through the deep learning model

def audio_to_tensor(FILE_PATH):
    
    # Once we have our file path, we need to get the waveform array (we get an extra frequency parameter, so we ignore that using indexing)

    waveform_arr = wavfile.read(FILE_PATH)[1]

    # From the line above we get an array that repersents the waveform of the audio file, we then use the following line to convert said waveform to a tensor

    waveform = torch.from_numpy(waveform_arr)
    
    return waveform.view(10584064).float()

def get_label(FILE_PATH):

    # Even though we have our tensor, we also need the label for the tensor
    # We want to return a tuple: Tensor, Label

    # Thankfully, our labels are simply on our file path!

    label = -1

    if 'Ariana_Grande' in FILE_PATH:
        label = torch.tensor([1, 0, 0], dtype=torch.float32).clone().detach()

    elif 'Kendrick_Lamar' in FILE_PATH:
        label =  torch.tensor([0, 1, 0], dtype=torch.float32).clone().detach()

    elif 'Travis_Scott' in FILE_PATH:
        label =  torch.tensor([0, 0, 1], dtype=torch.float32).clone().detach()
    
    return label

# Create trainset (array with data and labels)
trainset_x = []
trainset_y = []

directory = os.path.join(os.getcwd(), 'audio')

for dir in os.listdir(directory):
    for filename in os.listdir(os.path.join(directory, dir)):
        f = os.path.join(directory, dir, filename)
        if os.path.isfile(f):
            trainset_x.append(audio_to_tensor(f))
            trainset_y.append(get_label(f))

# Shuffle the trainsets
random.shuffle(trainset_x)
random.shuffle(trainset_y)

# Run the network
EPOCHS = 5
input_size = 10584064
hidden_size = 64
output_size = 3 

model = Net(input_size, hidden_size, output_size)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), 0.00000001)

for epoch in range(EPOCHS):

    print(f'EPOCH: {epoch} STARTING')

    for i in range(len(trainset_x)):

        outputs = model(trainset_x[i])
        loss = loss_fn(outputs, trainset_y[i])

        optim.zero_grad()
        loss.backward()

        optim.step()

        print(loss)
