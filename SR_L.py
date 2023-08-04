# installed librosa using pip install librosa
# installed pip install librosa soundfile numpy sklearn pyaudio

import librosa # for analyzing audio and music
import soundfile # This library is used to read and write audio files.
import os, glob, pickle 

# OS - This library is used to access the file system. 
#Glob - This library is used to search for files in a directory.
#Pickle - This library is used to save and load Python objects.

import numpy as np #This library is used to work with arrays.
from sklearn.model_selection import train_test_split #  used to split data into training and test sets.
from sklearn.neural_network import MLPClassifier # used to create and train neural networks
from sklearn.metrics import accuracy_score # used to evaluate the performance of a model.

# The script first loads the audio files from the `data` directory. 
# The audio files are divided into two classes: `speech` and `music`. 
# The script then extracts features from the audio files, 
# such as the mean and standard deviation of the audio signal. 
# The features are then used to train a neural network. 
# The neural network is trained on the training set and then evaluated on the test set. 
# The accuracy of the neural network is then reported.

#Define a function extract_feature to extract the mfcc, chroma, and mel features from a sound file. This function takes 4 parameters- the file name and three Boolean parameters for the three features:
# mfcc: Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound
# chroma: Pertains to the 12 different pitch classes
# mel: Mel Spectrogram Frequency

#DataFlair - Extract features (mfcc, chroma, mel) from a sound file

def extract_feature(file_name, mfcc,chroma,mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
            result = np.hstack((result,mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S= stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result,chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result = np.hstack((result,mel))
    return result

#Defining Dictionary (Emotions in the RAVDESS dataset)

emotions = {'01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#Emotion to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

#loading th Data (function: load_data() and extract features for each sound file)

def load_data(test_size = 0.2):
    x = []
    y = []
    for file in glob.glob("C:\\KD\\Python Learning\\Projects\\SR_P\\SR_D\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True,chroma=True,mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x),y,test_size=test_size,random_state=9 )

#Train and Split Data
x_train,x_test,y_train,y_test=load_data(test_size=0.25)

#Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

#Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

#Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, 
                    hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

#Train the model
model.fit(x_train,y_train)

#Predict for the test set
y_pred=model.predict(x_test)

#Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#rint the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))



