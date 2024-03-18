# sectors

![Descrição da Imagem](images/streamlit.png)


## Este é um projeto de Processamento de Linguagem Natural onde o interesse é, dado um texto de input, predizer se ele é referente à um ou mais setores dentre 5. 

**Os 5 setores são:**

```
1. Educação
2. Finanças
3. Indústrias
4. Órgão Público
5. Varejo
```

O Texto pode se enquadrar em um ou mais setores, como no exemplo "Prefeitura abre vagas para aulas de LIBRAS" onde o texto pode ser classificado em Órgão Público e Educação.

## Resumo:

O dataset utilizado possui a variável 'sentence' referente aos textos e a variável 'category' referente aos setores de interesse. Os textos estão disponíveis em português com as respectivas acentuações das palavras, com isso foi necessário realizar todo o pre processamento, que envolve desde a remoção de caracteres especiais até o padding (vide `analysis.ipynb`). Para o problema de multi label, foi decidido criar uma arquitetura de Rede Convolucional para as labels binarizadas, ou seja, foi utilizado a função de ativação sigmóide na última camada densa, onde o peso entre 0 e 1 atribuído à cada classe é comparado à um threshold. Se uma ou mais classe ultrapassar esse threshold, o output será de que o texto é referente à esses setores. 

Para mais informações sobre a abordagem [Large-scale multi-label text classification](https://keras.io/examples/nlp/multi_label_classification/)    

## Dataset:
  
- O dataset consiste em uma coluna `sentence`, referente aos textos a serem analisados, e a coluna `category` que contém os setores para o respectivo texto, podendo ser apenas um ou mais setores.

## Conteúdo:

**No repositório podem ser encontrados os arquivos:**
- `analysis.ipynb` - > Este notebook contém todo o pipeline da resolução do projeto, executado etapa por etapa.
- `NLPPipeline.py` -> Neste arquivo, é construída uma classe para execução do Pipeline de NLP (você pode passar seus próprio dataset).
- `run_pipeline.py` -> Este é um arquivo que exemplifica o uso da classe NLPPipeline.
- `NLPStreamlit.py` -> Esse arquivo constrói uma aplicação simples em Streamlit, onde o usuário passa uma frase e é retornada a classe do texto (ou classes).
- `model_tensorflowjs` -> Nesta pasta está o modelo convertido para java script (feito utilizando o tensorflowjs).
- `CNN_MultiLabel_NLP.h5` -> Este é o modelo serializado, útil para fazer o load caso não queira executar o treinamento.
- `dataset` -> Nesta pasta está o dataset utilizado para o projeto.
- `images` -> É uma pasta auxiliar para armazenar imagens do projeto.
- `requirements.txt` -> Dependências.

## Uso:

**Passos iniciais**
1. Clone este repositório para o seu computador.
   ```
   git clone https://github.com/MatheussAlvess/<nome_do_repositório>.git
   ```
3. Navegue até o diretório do projeto.
4. Garanta ter as dependências necessárias (vide `requirements.txt`)
   
- **Para realizar o reconhecimento de uma das 20 ações em um vídeo dado de input, execute o comando**

  ```
  python ActionDetection.py
  ```
  Garantindo que dentro do arquivo `ActionDetection.py` a classe "ACTIONS" seja instaciada com o nome do vídeo de interesse de input, assim o vídeo será processado e salvo com o nome `output_<nome_do_video>`. 
  
  > Ex.: Executando `python ActionDetection.py`, tendo instanciado "ACTIONS(video_name='libras.mp4')" dentro do arquivo, será salvo um vídeo nomeado `output_libras.mp4`.
  

- **Caso queira fazer o reconhecimento da ação em tempo real, basta executar:**

  ```
  python ActionDetection.py live
  ```
  
  Assim a webcam será aberta e poderá ser feito o reconhecimento em tempo real.
  
___________________________________________
  
## Para utilizar o projeto como base para um projeto próprio, realize as seguinte etapas:

**Uma vez que tudo esteja pronto para ser executado (repositório clonado):**

1. Armazene seus vídeos dentro de uma pasta, onde cada vídeo é nomeado de acordo com a ação, pois a classe é obtida a partir do nome do arquivo:
   Ex.: `data/libras.mp4`   
2. Execute o comando:
   ```
   python CreateDataset.py
   ```
   Dessa forma será criado o dataset de coordenadas a partir dos vídeos encontrados na pasta de referência. (Por _default_ é "data")
3. Execute o comando:
   ```
   python MLPModel.py
   ```
   Assim o modelo MLP será treinado com base no dataset de coordenadas. (A arquitetura e parâmetros podem ser modificados dentro do arquivo)
4. Por fim execute o comando para reconhecimento das ações:
   ```
   python ActionDetection.py
   ``` 

___________________________________________

#### Observações:

- Esta é a primeira versão do projeto, dessa forma, as classificações com base nas detecções podem não ser tão precisas para algumas ações.
  Isso se deve por alguns motivos, sendo alguns deles:
  1. Conjunto de dados relativamente pequeno: Considerei apenas um video curto para cada ação e as ações não variavam muito. Por exemplo, para aprender a ação 'paz'
     o modelo recebe um cenário onde uma mão está com 2 dedos levantados enquanto que a outra não está visível na imagem, logo, existe a associação de que quando um mão das mãos não está visível isso pode se configurar a ação 'paz',
     algo que não é necessariamente correto.
  
  2. Não houve um tratamento do dataset de coordenadas. Em alguns cenários a ação do sinal 'amigo' não tinha a detecção de nenhuma das mãos, o que faz com que o modelo entenda que a ausência das mão pode ser considerado a ação do sinal 'amigo'.
     
  3. Refinamento do modelo. O modelo MLP considerado não foi otimizado, em alguns cenários ele pode estar fazendo a associação dos valores das coordenadas com a ação que não necessariamente é a correta, como acontece para 'tchau', 'paz', 'telefone'.
     Por serem ações que tem as coordenadas muito parecidas, o modelo pode não ser robusto para identificar a classe.
     
  4. O ponto que, ao entendimento adiquirido durante a execução do projeto, mais impacta na confusão da classificação das ações é a falha de detecção dos landmarks.
     Como o modelo classificador depende das coordenadas, quando a detecção dos landmarks falham, o classificador fica perdido. E isso é mais grave no contexto de treinamento, pois o modelo pode estar aprendendo
     que a ausência de coordenadas pora as mãos é o sinal "telefone". Isso pode ser resolvido tatno com o tratamento dos dados, melhora na resolução do vídeo (facilitando a detectção) ou até mesmo considerar
     outro modelo para a detecção que seja mais eficiente.



> [!TIP]
> - Trabalhando em um outro projeto com MediaPipe, já tive experiência com o problema da falha de detecção de landmarks. Como alternativa, utilizei o Pose Estimation da YOLO, a qual é bem mais eficiente realizando as detecções (em troca de um maior custo computacional). [Utilizando YOLO para Pose Estimation](https://github.com/MatheussAlvess/Cervical_Posture_YOLO_Pose_Estimation).
> - O corpo do código de detecção de pose tem como principal referência esse repositório: [Body-Language-Decoder](https://github.com/nicknochnack/Body-Language-Decoder/tree/main).

