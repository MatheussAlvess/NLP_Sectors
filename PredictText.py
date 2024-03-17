import sys
from NLPPipeline import Pipeline 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict_sector(text: object = 'Texto com o tema de interesse', threashold: float = 0.3) -> list:

    pipeline = Pipeline()
    pipeline.processing_dataset()

    tokenizer = pipeline.tokenizer
    mlb = pipeline.mlb
    
    new_sentence = text

    # Pré-processamento da nova frase
    new_sentence_tokens = tokenizer.texts_to_sequences([new_sentence])
    new_sentence_tokens_padded = pad_sequences(new_sentence_tokens, padding='post', maxlen=100)

    # Previsão da nova frase
    predictions = model.predict(new_sentence_tokens_padded)

    # Decodificação da previsão
    threshold = 0.1  # Limiar de probabilidade para considerar a classe presente ou não
    predicted_labels = (predictions > threshold).astype(int)

    # Decodificando os rótulos previstos usando o MultiLabelBinarizer inverso
    predicted_categories = mlb.inverse_transform(predicted_labels)

    print('Setor(es) predito(s)\n',[i for i in predicted_categories[0]])
    return [i for i in predicted_categories[0]]


if __name__=='__main__':
    args = sys.argv
    model = load_model("CNN_MultiLabel_NLP.h5")

    texto = ' '.join(args)
    predict_sector(texto)