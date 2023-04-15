import torch
import os
from scipy import signal
from scipy.io import wavfile
import torch.nn as nn
import random


def spectrogram(FILE_PATH):
    sample_rate, samples = wavfile.read(FILE_PATH)
    _, _, spectrogram = signal.spectrogram(samples, sample_rate)

    return torch.tensor(spectrogram)

def get_label(FILE_PATH):

    # Even though we have our tensor, we also need the label for the tensor
    # We want to return a tuple: Tensor, Label

    # Thankfully, our labels are simply on our file path!

    label = -1

    if 'Ariana_Grande' in FILE_PATH:
        label = torch.tensor([1, 0], dtype=torch.long).clone().detach()

    elif 'Kendrick_Lamar' in FILE_PATH:
        label =  torch.tensor([0, 1], dtype=torch.long).clone().detach()

    #elif 'Travis_Scott' in FILE_PATH:
    #    label =  torch.tensor([0, 0, 1], dtype=torch.long).clone().detach()
    
    return label


def convert_audio_files(directory):
    # Create trainset (array with data and labels)
    trainset = []

    #directory = os.path.join(os.getcwd(), 'audio')

    for dir in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, dir)):
            f = os.path.join(directory, dir, filename)
            if os.path.isfile(f):
                trainset.append([spectrogram(f), get_label(f)])

    return trainset

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(5292032, 64, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (1, 1)),
            nn.ReLU(),
#            nn.Conv2d(64, 64, (1, 1)),
#            nn.ReLU(),
            nn.Conv2d(64, 2, (1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2, 2)
        )

    def forward(self, x):
        return self.network(x)

net = Net()
optim = torch.optim.Adam(net.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()


def train_model(trainset):
        for epoch in range(5):

            for batch in trainset:
                X, y = batch

                output = net(X)

                loss = criterion(output, y)

                optim.zero_grad()
                loss.backward()
                optim.step()

                print(loss)

        torch.save(net.state_dict(), 'net.pth')


def check(path):

    net.load_state_dict(torch.load('net.pth'))
    tensor = spectrogram(path)

    output = net(tensor)
    predicted_class = torch.argmax(output)
    
    if predicted_class.item() == 0:
        return 'Ariana Grande'

    elif predicted_class.item() == 1:
        return 'Kendrick Lamar'
    
    else:
        return 'Travis Scott'
    


trainset = convert_audio_files('audio')
random.shuffle(trainset)
train_model(trainset)

'''net.load_state_dict(torch.load('net.pth'))

net.eval()

print(trainset[1][1])

print(net(trainset[1][0]))'''

print(check('audio/Ariana_Grande/Ariana_Grande-3.wav'))