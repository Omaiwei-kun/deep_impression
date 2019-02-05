import cv2

import audiovisual_stream
import chainer.serializers
import librosa
import numpy
import skvideo.io
import numpy as np

FRAMES_LIMIT = 25


def load_audio(data):
    return librosa.load(data, 16000)[0][None, None, None, :]


def load_model():
    model = audiovisual_stream.ResNet18().to_gpu()

    chainer.serializers.load_npz('./model', model)

    return model


def predict_trait(data, model):
    # videoCapture = skvideo.io.vreader(data, num_frames=27)
    videoCapture = skvideo.io.vreader(data)

    audio_features = load_audio(data)

    x = []
    pred = []

    frames_count = 0
    for image in videoCapture:

        x.append(numpy.rollaxis(image, 2))

        frames_count += 1

        if frames_count == FRAMES_LIMIT:
            x = [audio_features, numpy.array(x, 'float32')]
            pred.append(model(x))
            frames_count = 0
            x = []

    return np.mean(np.asarray(pred), axis=0)
