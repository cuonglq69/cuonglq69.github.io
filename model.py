import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from music21 import converter, instrument, note, chord, stream
from sklearn.model_selection import train_test_split
import tensorflow
import keras.models
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Bidirectional, LSTM
from keras.optimizers import Adamax
from collections import Counter
import random
import matplotlib.patches as mpatches
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
# from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import os, sys


# A music score includes notes (single musical vibrations) and chords (combination of notes)
# Pick notes and chords from the midi files in the music_file folder
def collect_notes():
    notes = []

    files = glob.glob("midi_file_bethoven/*.mid")

    # random_files = random.sample(files, 3)  # Randomly select 5 files from the folder

    for file in files:

        midi = converter.parse(file)

        print("Push file %s" % file)

        picked_notes = None
        # If a file includes instrument parts
        try:
            track = instrument.partitionByInstrument(midi)
            picked_notes = track.parts[0].recurse()
        # If a file does not have instrument parts and instead has a flat structure containing notes directly
        except:
            picked_notes = midi.flat.notes

        for element in picked_notes:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes
def build_model(X,y):
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(128))
    model.add(Dense(128))
    model.add(Dropout(0.1))
    model.add(Dense(y.shape[1], activation='softmax'))
    opt = Adamax(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.summary()
    return model

def convert_to_midi(music):
    melody = []
    # Determines when the note will be played in relation to other notes/chords in the sequence.
    offset = 0
    for item in music:
        # If item is a chord, it is normally a sequence of number and dot, ex: 10.2.3
        if ("." in item or item.isdigit()):
            # Split notes in the chord
            chord_has_notes = item.split(".")
            notes = []
            for note_item in chord_has_notes:
                note_item_to_int=int(note_item )
                # note_set contain info of a note including pitch, waves...
                note_set = note.Note(note_item_to_int)
                notes.append(note_set)
                chord_set = chord.Chord(notes) # a collection of notes played simultaneously as a chord
                chord_set.offset = offset
                melody.append(chord_set)
        else: # If the pattern is a single note, It has standard format, like: E4
            #print(i)
            # If the pattern is a single note, It has standard format, like: E4
            note_set = note.Note(item)
            note_set.offset = offset
            melody.append(note_set)
        # increase offset each iteration so that notes do not stack
        offset += 1
    melody_midi = stream.Stream(melody)
    return melody_midi

def Music_creator(note_num, model, X_test, length, uniq_count, note_map_reverse):
    rand_sample = X_test[np.random.randint(0, len(X_test) - 1)]
    generated_music = ""
    new_notes = [] = []
    for i in range(note_num):
        rand_sample = rand_sample.reshape(1, length, 1)
        prediction = model.predict(rand_sample, verbose=0)[0]
        # print(prediction)
        # Applies a logarithm operation on the prediction and divides it by a value (1.0)
        # to control the diversity of the generated notes.
        # prediction = np.log(prediction) / 1.0  # diversity
        # # Exponentiates the predicted values to reverse the logarithm operation.
        # expo_preds = np.exp(prediction)
        # # Normalizes the prediction probabilities to make them sum up to 1
        # prediction = expo_preds / np.sum(expo_preds)
        # Selects the index of the note with the highest probability as the next generated note
        index = np.argmax(prediction)
        # Converts the index to a normalized value between 0 and 1
        norm_index = index / float(uniq_count)
        # Appends the index of the new generated note to the new_notes
        new_notes.append(index)
        # Converts the new note indexes back into their corresponding labels and stores them in the generated_music list.
        generated_music = [note_map_reverse[char] for char in new_notes]
        # Inserts the normalized index value at the end of the random sample array
        rand_sample = np.insert(rand_sample[0], len(rand_sample[0]), norm_index)
        # Removes the first element from the random sample array, shifting the remaining elements
        rand_sample = rand_sample[1:]

    generated_melody = convert_to_midi(generated_music)
    music_midi = stream.Stream(generated_melody)
    return music_midi

def save_model(model, filepath):
    model.save(filepath)
def load_model(filepath):
    return keras.models.load_model(filepath)

def run():
    note_collected = collect_notes()
    print("All notes in all music files:", len(note_collected))
    count_freq = Counter(note_collected)

    # Define infrequent notes and chords
    infreq_note = []
    for id, (k, value) in enumerate(count_freq.items()):
        if value < 100:
            in_note = k
            infreq_note.append(in_note)

    # Removing infrequent notes
    for j in note_collected:
        if j in infreq_note:
            note_collected.remove(j)
    print("All notes after removing infrequent notes:", len(note_collected))

    # Create a list of unique notes
    uniq_note = sorted(list(set(note_collected)))

    # length of all notes
    notes_count = len(note_collected)
    # length of unique notes
    uniq_count = len(uniq_note)

    # Create a dictionary to label notes with indexes and reverse it
    note_map = dict((note, id) for id, note in enumerate(uniq_note))
    note_map_reverse = dict((id, note) for id, note in enumerate(uniq_note))

    print("Number of notes:", notes_count)
    print("Number of unique notes:", uniq_count)

    # Create input sequences for the network and their respective outputs
    length = 40
    features = []
    targets = []
    for i in range(0, notes_count - length, 1):
        feature = note_collected[i:i + length]
        target = note_collected[i + length]
        features.append([note_map[j] for j in feature])
        targets.append(note_map[target])

    sequence_count = len(targets)
    print("Number of sequences in dataset:", sequence_count)

    # Reshape and normalize input
    X = (np.reshape(features, (sequence_count, length, 1))) / float(uniq_count)
    # Encode the output
    y = tensorflow.keras.utils.to_categorical(targets)

    # Split data to train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Build the Model
    model = build_model(X,y)
    # Train the Model
    result = model.fit(X_train, y_train, batch_size=68, epochs=200)
    save_model(model, 'model_bethoven')

    # Visualize the loss in learning
    # %matplotlib inline
    result_df = pd.DataFrame(result.history)
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Loss Distribution in Learning")
    pl = sns.lineplot(data=result_df["loss"], color="#444160")
    pl.set(ylabel="Training Loss")
    pl.set(xlabel="Epochs")
    plt.show()

    music_midi = Music_creator(100, model, X_test, length, uniq_count, note_map_reverse)

    # To save the generated melody
    music_midi.write('midi', 'song1.mid')
    return 'song1.mid'

def run2():
    note_collected = collect_notes()
    print("All notes in all music files:", len(note_collected))
    count_freq = Counter(note_collected)

    # Define infrequent notes and chords
    infreq_note = []
    for id, (k, value) in enumerate(count_freq.items()):
        if value < 100:
            in_note = k
            infreq_note.append(in_note)

    # Removing infrequent notes
    for j in note_collected:
        if j in infreq_note:
            note_collected.remove(j)
    print("All notes after removing infrequent notes:", len(note_collected))

    # Create a list of unique notes
    uniq_note = sorted(list(set(note_collected)))

    # length of all notes
    notes_count = len(note_collected)
    # length of unique notes
    uniq_count = len(uniq_note)

    # Create a dictionary to label notes with indexes and reverse it
    note_map = dict((note, id) for id, note in enumerate(uniq_note))
    note_map_reverse = dict((id, note) for id, note in enumerate(uniq_note))

    print("Number of notes:", notes_count)
    print("Number of unique notes:", uniq_count)

    # Create input sequences for the network and their respective outputs
    length = 40
    features = []
    targets = []
    for i in range(0, notes_count - length, 1):
        feature = note_collected[i:i + length]
        target = note_collected[i + length]
        features.append([note_map[j] for j in feature])
        targets.append(note_map[target])

    sequence_count = len(targets)
    print("Number of sequences in dataset:", sequence_count)

    # Reshape and normalize input
    X = (np.reshape(features, (sequence_count, length, 1))) / float(uniq_count)
    # Encode the output
    y = tensorflow.keras.utils.to_categorical(targets)

    # Split data to train and test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Load the Model

    model = load_model('model_bethoven')

    music_midi = Music_creator(100, model, X, length, uniq_count, note_map_reverse)

    # To save the generated melody
    music_midi.write('midi', 'static/song2.mid')
    return 'song2.mid'