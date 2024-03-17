import os 
import re
import string
import spacy as sp
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ['PYTHONHASHSEED'] = str(42) 
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISM'] = '1'


class Pipeline():
    """
    Classe que encapsula o pipeline de processamento e previsão de um modelo CNN multilabel para classificação de texto.

    Attributes:
        dataset_path (str): Caminho para o diretório contendo o conjunto de dados.
        dataset_name (str): Nome do arquivo CSV contendo o conjunto de dados.
        sentences_variable (str): Nome da variável que contém os textos no conjunto de dados.
        categories_variable (str): Nome da variável que contém as categorias no conjunto de dados.
        model_name (str): Nome do arquivo para salvar o modelo treinado.
        save (bool): Indica se o modelo treinado deve ser salvo após o treinamento.
        emb_dim (int): Dimensão do espaço de incorporação (embedding) dos tokens.
        nb_filters (int): Número de filtros para as camadas convolucionais do modelo CNN.
        batch_size (int): Tamanho do lote utilizado durante o treinamento do modelo.
        ffn_units (int): Número de unidades para as camadas densas (fully connected) do modelo CNN.
        nb_classes (int): Número de classes de saída do modelo.
        dropout_rate (float): Taxa de dropout para a regularização do modelo.
        nb_epochs (int): Número de épocas de treinamento do modelo.
        verbose (int): Nível de detalhes das mensagens durante o treinamento do modelo.

    Methods:
        cleaning(text, nlp, stop_words): Realiza a limpeza de um texto, incluindo conversão para minúsculas, remoção de caracteres especiais, stopwords e pontuações.
        processing_dataset(): Processa o conjunto de dados, incluindo carregamento, limpeza dos textos, tokenização, binarização das categorias e divisão em conjuntos de treino e teste.
        DCNN(vocab_size, training=False, name='dcnn'): Define o modelo CNN com arquitetura especificada.
        run_model(): Treina o modelo CNN utilizando os dados de treino e avalia o desempenho com os dados de teste.
        predict_sector(text='Texto com o tema de interesse', threshold=0.3): Realiza previsões de categorias para um texto de entrada utilizando o modelo treinado.

    Para o passo a passo, acesse o notebook "analisys.ipynb" neste mesmo repositório.
    """

    def __init__(self,
                dataset_path: object = 'dataset/',
                dataset_name: object = 'dataset.csv',
                sentences_variable: object = 'sentence',
                categories_variable: object = 'category',
                model_name: object = 'CNN_Multilabel_NLP.keras',
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


    # Limpeza dos textos (tudo em minúsculo, removendo acentos, caracteres especiais e stopwords)
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

        text = text.lower()
        document = nlp(text)

        words = []
        for token in document:
            words.append(token.text)

        words = [word for word in words if word not in stop_words and word not in string.punctuation] 
        words = ' '.join([str(element) for element in words])

        return words

    # Função para processamento do conjunto de dados (Desde a limpeza até a tokenização e padding)
    def processing_dataset(self):
        print('processing dataset')
        dataset_path = self.dataset_path 
        dataset_name = self.dataset_name

        self.dataset = pd.read_csv(os.path.join(dataset_path, dataset_name))
        
        self.dataset[self.categories_variable] = [set.split(',') for set in self.dataset[self.categories_variable]]

        nlp = sp.blank("pt")
        stop_words = sp.lang.pt.STOP_WORDS

        self.dataset[self.sentences_variable] = [self.cleaning(text,nlp,stop_words) for text in self.dataset[self.sentences_variable]]

        X = self.dataset[self.sentences_variable]

        print('Tokenizing')
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(X.tolist())
        self.data_inputs = self.tokenizer.texts_to_sequences(X.tolist())

        self.vocab_size = len(self.tokenizer.word_index) + 1
        max_len = max([len(sentence) for sentence in self.data_inputs])+20

        self.data_inputs = tf.keras.preprocessing.sequence.pad_sequences(self.data_inputs,
                                                            value = 0,
                                                            padding = 'post',
                                                            maxlen=max_len)

        
        self.mlb = MultiLabelBinarizer()
        self.data_labels = self.mlb.fit_transform(self.dataset['category'])

        self.train_inputs, self.test_inputs, self.train_labels, self.test_labels = train_test_split(self.data_inputs,
                                                                                            self.data_labels,
                                                                                            test_size=0.1,
                                                                                            stratify = self.data_labels)
                    
        print('Shape dos dados de treinamento:',f'Inputs: {self.train_inputs.shape} Labels: {self.train_labels.shape}')
        print('Shape dos dados de teste:',f'Inputs: {self.test_inputs.shape} Labels: {self.test_labels.shape}')

    # Modelo CNN 
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
        outputs = tf.keras.layers.Dense(units=self.nb_classes, activation='softmax', name='output')(dropout)

        # Criando o modelo
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

        return model
    

    # Executa o treinamento da CNN com os textos processados como input
    def run_model(self):
        print('Training started')
        tf.random.set_seed(1)
        ## Parametros
        
        ## Inicializando modelo
        model = self.DCNN(vocab_size=self.vocab_size)
        
        ## Compilando
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

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


    # Realiza as predições de novos textos
    def predict_sector(self,text: object = 'Texto com o tema de interesse', threshold: float = 0.3) -> list:
        
        new_sentence = text

        # Pré-processamento da nova frase
        new_sentence_tokens = self.tokenizer.texts_to_sequences([new_sentence])
        new_sentence_tokens_padded = pad_sequences(new_sentence_tokens, padding='post', maxlen=100)

        # Previsão da nova frase
        predictions = self.model.predict(new_sentence_tokens_padded)

        # Decodificação da previsão
        threshold = 0.1  # Limiar de probabilidade para considerar a classe presente ou não
        predicted_labels = (predictions > threshold).astype(int)

        # Decodificando os rótulos previstos usando o MultiLabelBinarizer inverso
        predicted_categories = self.mlb.inverse_transform(predicted_labels)

        print([i for i in predicted_categories[0]])
        return [i for i in predicted_categories[0]]
        
    
if __name__=='__main__':

    pipeline = Pipeline(nb_epochs=100)
    pipeline.processing_dataset()
    pipeline.run_model()
    pipeline.predict_sector('Estude LIBRAS')