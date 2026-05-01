import os
import numpy as np
import pickle as pkl
import tensorflow as tf
import re
from functools import lru_cache
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, LSTM, Bidirectional, TimeDistributed
from tensorflow.keras.initializers import glorot_normal

class ShakkalaDiacritizationPipeline:
    def __init__(self, constants_path="shakkala_const", model_weight_name="avg_20_fixed.h5", use_gpu=True):
        self.CONSTANTS_PATH = constants_path
        self.model_weight = f"{self.CONSTANTS_PATH}/{model_weight_name}"
        self.use_gpu = use_gpu
        
        # Load Resources
        self._load_constants()
        
        # Initialize Model
        if self.use_gpu:
            with tf.device('/GPU:0'):
                self.model = self.create_model()
                self.model.load_weights(self.model_weight)
        else:
            self.model = self.create_model()
            self.model.load_weights(self.model_weight)

    def _load_constants(self):
        with open(self.CONSTANTS_PATH + '/ARABIC_LETTERS_LIST.pickle', 'rb') as file:
            self.ARABIC_LETTERS_LIST = pkl.load(file)

        with open(self.CONSTANTS_PATH + '/DIACRITICS_LIST.pickle', 'rb') as file:
            self.DIACRITICS_LIST = pkl.load(file)

        with open(self.CONSTANTS_PATH + '/RNN_CLASSES_MAPPING.pickle', 'rb') as file:
            self.CLASSES_MAPPING = pkl.load(file)

        with open(self.CONSTANTS_PATH + '/RNN_BIG_CHARACTERS_MAPPING.pickle', 'rb') as file:
            self.CHARACTERS_MAPPING = pkl.load(file)

        with open(self.CONSTANTS_PATH + '/RNN_REV_CLASSES_MAPPING.pickle', 'rb') as file:
            self.REV_CLASSES_MAPPING = pkl.load(file)

    def create_model(self):
        # if tf.test.is_gpu_available():
        #     SelectedLSTM = CuDNNLSTM
        # else:
        #     SelectedLSTM = LSTM
        SelectedLSTM = LSTM

        inputs = Input(shape=(None,))

        embeddings = Embedding(input_dim=len(self.CHARACTERS_MAPPING), # Input berupa semua karakter arab yang mungkin
                              output_dim=25,
                              embeddings_initializer=glorot_normal(seed=961))(inputs)

        blstm1 = Bidirectional(SelectedLSTM(units=256,
                                        return_sequences=True,
                                        kernel_initializer=glorot_normal(seed=961)))(embeddings)
        dropout1 = Dropout(0.5)(blstm1)
        blstm2 = Bidirectional(SelectedLSTM(units=256,
                                        return_sequences=True,
                                        kernel_initializer=glorot_normal(seed=961)))(dropout1)
        dropout2 = Dropout(0.5)(blstm2)

        dense1 = TimeDistributed(Dense(units=512,
                                      activation='relu',
                                      kernel_initializer=glorot_normal(seed=961)))(dropout2)
        dense2 = TimeDistributed(Dense(units=512,
                                      activation='relu',
                                      kernel_initializer=glorot_normal(seed=961)))(dense1)

        output = TimeDistributed(Dense(units=len(self.CLASSES_MAPPING), # Outputnya di Mapping pada label diakritik
                                      activation='softmax',
                                      kernel_initializer=glorot_normal(seed=961)))(dense2)

        model = Model(inputs, output)

        # SOLUSI: Ganti dengan Adam standar.
        # NormalizedOptimizer hanya penting saat fase latihan (training).
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        return model
    def remove_diacritics(self, data):
        return data.translate(str.maketrans('', '', ''.join(self.DIACRITICS_LIST)))

    def predict_rnn(self, text):
        lines = text.split('\n')
        result = ''

        for line in lines:
            line = self.filter_arabic_only(line)
            x = [self.CHARACTERS_MAPPING['<SOS>']]
            for idx, char in enumerate(line):
                if char in self.DIACRITICS_LIST:
                    continue
                if char not in self.CHARACTERS_MAPPING:
                    x.append(self.CHARACTERS_MAPPING['<UNK>'])
                else:
                    x.append(self.CHARACTERS_MAPPING[char])
            x.append(self.CHARACTERS_MAPPING['<EOS>'])
            x = np.array(x).reshape(1, -1)

            predictions = self.model.predict(x).squeeze()
            predictions = predictions[1:]
            output = ''
            for char, prediction in zip(self.remove_diacritics(line), predictions):
                output += char

                if char not in self.ARABIC_LETTERS_LIST:
                    continue

                prediction = np.argmax(prediction)

                if '<' in self.REV_CLASSES_MAPPING[prediction]:
                    continue

                output += self.REV_CLASSES_MAPPING[prediction]
            
            result += output + '\n'

        return result
    
    def filter_arabic_only(self, text):
        # Pattern untuk mencakup seluruh blok karakter Arab dan diakritiknya
        arabic_pattern = re.compile(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\ufb50-\ufdff\ufe70-\ufefc\s]')
        
        filtered_text = arabic_pattern.sub('', text)
        filtered_text = ' '.join(filtered_text.split())
    
        return filtered_text
# --- Module-level API ---

@lru_cache(maxsize=1)
def get_pipeline() -> ShakkalaDiacritizationPipeline:
    """Singleton pipeline instance (safe to use as a FastAPI dependency)."""
    return ShakkalaDiacritizationPipeline()


def diacritic_text(text: str) -> str:
    """Diacritize Arabic text — the public entry point used by the API."""
    return get_pipeline().predict_rnn(text)