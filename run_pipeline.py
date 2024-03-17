
from NLPPipeline import Pipeline 

params = {
    'dataset_path': 'dataset/',
    'dataset_name': 'dataset.csv',
    'sentences_variable': 'sentence',
    'categories_variable': 'category',
    'model_name': 'cnn_model.keras',
    'save': True,
    'emb_dim': 128,
    'nb_filters': 50,
    'ffn_units': 256,
    'nb_classes': 5,
    'batch_size': 64,
    'dropout_rate': 0.2,
    'nb_epochs': 100,
    'verbose': 1}


pipeline = Pipeline(**params)
pipeline.processing_dataset()
pipeline.run_model()
pipeline.predict_sector('As aulas nas escolas começam hoje!')