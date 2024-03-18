
import streamlit as st
from NLPPipeline import Pipeline 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("CNN_MultiLabel_NLP.h5")

def predict_sector(model, text, threshold: float = 0.3) -> list:

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
    predicted_labels = (predictions > threshold).astype(int)

    # Decodificando os rótulos previstos usando o MultiLabelBinarizer inverso
    predicted_categories = mlb.inverse_transform(predicted_labels)

    
    return [i for i in predicted_categories[0]]


def main():
    st.title("Classificação de Setor")

    st.subheader("Insira um texto:")

    # Adiciona um campo de texto para receber o input do usuário
    texto_input = st.text_input("")
    st.markdown(texto_input)

    # Se for inserido um texto é predito o setor
    if texto_input:
        st.write("Setor(es) predito(s):", predict_sector(model,texto_input))

    

if __name__ == "__main__":
    main()
