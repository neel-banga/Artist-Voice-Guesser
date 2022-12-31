import numpy as np
import os
import scipy.io.wavfile as wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class VoiceDetection(nn.Module):

    def __init__(self):
        # First let's initialize nn.module; super corresponds with the parent class (we defined up top), then from that parent class we simply run the __init__ function.
        super().__init__()
        
        # Here let's define our layers, in our __init__ method we define the fully connected layers, then in other methods we control how data passes through
        # BTW: fc = "fully connected"
        
        self.fc1 = nn.Linear(11520000, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 64)
        self.fc8 = nn.Linear(64, 3)

        # At the end we output 3 as we have ten possible anwers (0-2) [These repersent the three artists]

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)        

        return F.log_softmax(x, dim=1)

# Here we convert our audio file to a tensor; we do this so we can pass the audio tensors through the deep learning model

def audio_to_tensor(FILE_PATH):
    
    # Once we have our file path, we need to get the waveform array (we get an extra frequency parameter, so we ignore that using indexing)

    waveform_arr = wavfile.read(FILE_PATH)[1]

    # From the line above we get an array that repersents the waveform of the audio file, we then use the following line to convert said waveform to a tensor

    waveform = torch.from_numpy(waveform_arr).float()

    # Even though we have our tensor, we also need the label for the tensor
    # We want to return a tuple: Tensor, Label

    # Thankfully, our labels are simply on our file path!

    label = -1

    if 'Ariana_Grande' in FILE_PATH:
        label = 0

    elif 'Kendrick_Lamar' in FILE_PATH:
        label =  1
    
    elif 'Travis_Scott' in FILE_PATH:
        label =  2

    return waveform.view(11520000), label



#print(audio_to_tensor('/run/media/neel/Storage/Code/programs/ArtistVoice/audio/Kendrick_Lamar/Kendrick_Lamar0.wav'))


# Create trainset (array with data and labels)
trainset = []

directory = '/run/media/neel/Storage/Code/programs/ArtistVoice/audio/'

for dir in os.listdir(directory):
    print(dir)
    for filename in os.listdir(os.path.join(directory, dir)):
        print(filename)
        f = os.path.join(directory, dir, filename)
        if os.path.isfile(f):
            print(audio_to_tensor(f))
            trainset.append(audio_to_tensor(f))

print(trainset)

'''
voice_det = VoiceDetection()

optimizer = optim.Adam(voice_det.parameters(), lr=0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        # data is a batch of featuresets and label
        x, y = data
        neural_network1.zero_grad()
        output = voice_det(x.view(-1, 784))
        loss = F.nll_loss(output, y) # Use nll_loss if you have one value as your output (in data) and use msq if it's a vector
        loss.backward()
        optimizer.step()

    print(loss)
'''
