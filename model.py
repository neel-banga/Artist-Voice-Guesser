import numpy as np
import os
import scipy.io.wavfile as wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Create the network



# Here we convert our audio file to a tensor; we do this so we can pass the audio tensors through the deep learning model

def audio_to_tensor(FILE_PATH):
    
    # Once we have our file path, we need to get the waveform array (we get an extra frequency parameter, so we ignore that using indexing)

    waveform_arr = wavfile.read(FILE_PATH)[1]

    # From the line above we get an array that repersents the waveform of the audio file, we then use the following line to convert said waveform to a tensor

    waveform = torch.from_numpy(waveform_arr)#.float()

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

    return waveform.view(10584064), torch.tensor(label)



#print(audio_to_tensor('/run/media/neel/Storage/Code/programs/ArtistVoice/audio/Kendrick_Lamar/Kendrick_Lamar0.wav'))


# Create trainset (array with data and labels)
trainset = []

directory = '/run/media/neel/Storage/Code/programs/ArtistVoice/audio/'

for dir in os.listdir(directory):
    for filename in os.listdir(os.path.join(directory, dir)):
        f = os.path.join(directory, dir, filename)
        if os.path.isfile(f):
            trainset.append(audio_to_tensor(f))

print(trainset)

# Run the network
