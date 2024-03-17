import os 
import re
import string
import numpy as np
import spacy as sp
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Pipeline():
    def __init__(self,
                dataset_path: object = 'dataset/',
                dataset_name: object = 'dataset.csv',
                sentences_variable: object = 'sentence',
                categories_variable: object = 'category',
                model_name: object = 'cnn_model.keras',
                save: bool = True,
                emb_dim: int = 128,
                nb_filters: int = 100,
                batch_size: int = 32,
                ffn_units: int = 512,
                nb_classes: int = 5,
                dropout_rate: float = 0.2,
                nb_epochs: int = 100,
                verbose: int = 1,
                ) -> None:

        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.sentences_variable = sentences_variable
        self.categories_variable = categories_variable
        self.model_name = model_name
        self.save = save
        self.emb_dim = emb_dim
        self.nb_filters = nb_filters
        self.ffn_units = ffn_units
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.nb_epochs = nb_epochs
        self.verbose = verbose


    def cleaning(self,text,nlp,stop_words):
        text = text.lower()

        text = re.sub(r"[!@#%&$*º“”—-]", "", text)
        text = re.sub(r"[àáâãäåã]", "a", text)
        text = re.sub(r"[èéêë]", "e", text)
        text = re.sub(r"[ìíîï]", "i", text)
        text = re.sub(r"[òóôõö]", "o", text)
        text = re.sub(r"[ùúûü]", "u", text)
        text = re.sub(r"[ýÿ]", "y", text)
        text = re.sub(r"[ç]", "c", text)
        text = re.sub(r"[0-9]", "", text)

        text = text.lower()
        document = nlp(text)

        words = []
        for token in document:
            words.append(token.text)

        words = [word for word in words if word not in stop_words and word not in string.punctuation] 
        words = ' '.join([str(element) for element in words])

        return words

    def processing_dataset(self):
        print('processing dataset')
        dataset_path = self.dataset_path 
        dataset_name = self.dataset_name

        self.dataset = pd.read_csv(os.path.join(dataset_path, dataset_name))
        
        self.dataset[self.categories_variable] = [set.split(',') for set in self.dataset[self.categories_variable]]
        self.dataset = self.dataset.explode(self.categories_variable).reset_index(drop=True)

        nlp = sp.blank("pt")
        stop_words = sp.lang.pt.STOP_WORDS

        self.dataset[self.sentences_variable] = [self.cleaning(text,nlp,stop_words) for text in self.dataset[self.sentences_variable]]
        self.dataset[self.categories_variable] = [self.cleaning(text,nlp,stop_words) for text in self.dataset[self.categories_variable]]

        self.enc = LabelEncoder()
        
        X = self.dataset[self.sentences_variable]
        y = self.enc.fit_transform(self.dataset[self.categories_variable])
        self.data_labels = np.array(y)

        print('Tokenizing')
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(X, target_vocab_size=10000)
        self.data_inputs = [self.tokenizer.encode(sentence) for sentence in X]

        max_len = max([len(sentence) for sentence in self.data_inputs])+20

        self.data_inputs = tf.keras.preprocessing.sequence.pad_sequences(self.data_inputs,
                                                            value = 0,
                                                            padding = 'post',
                                                            maxlen=max_len)

        
        self.train_inputs, self.test_inputs, self.train_labels, self.test_labels = train_test_split(self.data_inputs,
                                                                                self.data_labels,
                                                                                test_size=0.1,
                                                                                stratify = self.data_labels)
        
        print('Shape dos dados de treinamento:',f'Inputs: {self.train_inputs.shape} Labels: {self.train_labels.shape}')
        print('Shape dos dados de teste:',f'Inputs: {self.test_inputs.shape} Labels: {self.test_labels.shape}')

    def DCNN(self,
            vocab_size,
            training=False,
            name='dcnn'):

        # Definindo a entrada do modelo
        inputs = tf.keras.Input(shape=(None,), name='input_text')

        # Camada de embeddings
        x = tf.keras.layers.Embedding(vocab_size, self.emb_dim)(inputs)

        # Convoluções de Bigrama seguidas por pooling
        conv1 = tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=2, padding='same', activation='relu')(x)
        conv1 = tf.keras.layers.GlobalMaxPool1D()(conv1)

        # Convoluções de Trigrama seguidas por pooling
        conv2 = tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=3, padding='same', activation='relu')(x)
        conv2 = tf.keras.layers.GlobalMaxPool1D()(conv2)

        # Convoluções de Trigrama seguidas por pooling
        conv3 = tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=3, padding='same', activation='relu')(x)
        conv3 = tf.keras.layers.GlobalMaxPool1D()(conv3)

        # Convoluções de Quatrigrama seguidas por pooling
        conv4 = tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=4, padding='same', activation='relu')(x)
        conv4 = tf.keras.layers.GlobalMaxPool1D()(conv4)

        # Convoluções de Quatrigrama seguidas por pooling
        conv5 = tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=4, padding='same', activation='relu')(x)
        conv5 = tf.keras.layers.GlobalMaxPool1D()(conv5)

        # Concatenando as saídas das convoluções
        merged = tf.keras.layers.concatenate([conv1, conv2, conv3, conv4, conv5], axis=-1)

        # Rede densa
        dense = tf.keras.layers.Dense(units=self.ffn_units, activation='relu')(merged) # Camada densa inicial
        dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)(dense, training=training) # Camada de dropout
        
        # Camada de saída
        if self.nb_classes == 2:
            outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', name='output')(dropout)
        else:
            outputs = tf.keras.layers.Dense(units=self.nb_classes, activation='softmax', name='output')(dropout)

        # Criando o modelo
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

        return model
    
    def run_model(self):
        print('Training started')
        ## Parametros
        vocab_size = self.tokenizer.vocab_size

        ## Inicializando modelo
        model = self.DCNN(vocab_size=vocab_size)
        
        ## Compilando
        if self.nb_classes == 2:
            model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        else:
            model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

        self.history = model.fit(self.train_inputs, self.train_labels,
                            batch_size = self.batch_size,
                            epochs = self.nb_epochs,
                            verbose = 1,
                            validation_split = .1)

        self.results = model.evaluate(self.test_inputs, self.test_labels, batch_size=self.batch_size)
        print('loss: ',self.results[0],'\nacurácia: ',self.results[1])

        if self.save:
            ## Salvando o modelo 
            model.save(f'{self.model_name}')
        
        self.model = model

        
    def predict_sector(self,text: object = 'Texto com o tema de interesse', threashold: float = 0.2) -> list:
        # Tokeniza o texto usando o tokenizador
        text = self.tokenizer.encode(text)
        
        # Faz a previsão usando o modelo
        list_of_predictions = self.model.predict(np.array([text]))
        
        # Encontra os índices das previsões que têm uma probabilidade acima do threashold especificado
        ind = []
        for i in range(len(list_of_predictions)):
            ind.append(np.where(list_of_predictions[i] > threashold)[0].tolist())
            
        lab = []
        # Converte os índices das previsões em rótulos
        for i in ind:
            # Se houver mais de um índice com probabilidade acima do limiar
            if len(i)>1:
                aux = []
                # Converte cada índice em seu rótulo correspondente
                for j in i:
                    aux.append(self.enc.inverse_transform([j]).tolist()[0])
                lab.append(aux)
            # Se houver apenas um índice com probabilidade acima do limiar
            else:
                lab.append(self.enc.inverse_transform(i).tolist()[0])

        print(lab) 
        return lab

    
if __name__=='__main__':

    pipeline = Pipeline(nb_epochs=100)
    pipeline.processing_dataset()
    pipeline.run_model()
    pipeline.predict_sector('Estude LIBRAS')