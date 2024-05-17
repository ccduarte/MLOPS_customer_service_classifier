# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_output_filepath', type=click.Path())
def main(input_filepath, model_output_filepath):
    """ Runs training and evaluation scripts to train a model on the processed data and
        evaluate its performance.
    """
    logger = logging.getLogger(__name__)
    logger.info('⌛ Iniciando o treinamento do modelo...')

    # Carregar o arquivo de features
    try:
        df = pd.read_csv(input_filepath)
        logger.info('✅ Arquivo de features carregado com sucesso!')
    except pd.errors.ParserError as e:
        logger.error(f'❌ Erro ao ler o arquivo CSV de features: {e}')
        return

    # Garantir que a coluna 'categoria' está presente
    if 'categoria' not in df.columns:
        logger.error('❌ A coluna "categoria" não foi encontrada no arquivo de features.')
        return

    # Remover linhas com valores NaN
    df = df.dropna()
    logger.info(f'✅ DataFrame sem NaNs: {df.shape[0]} linhas restantes.')

    # Dividir os dados em conjuntos de treino e teste
    x = df.drop(columns=['categoria'])
    y = df['categoria']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    # Treinamento do modelo
    model = LogisticRegression(random_state=42)
    model.fit(x_train, y_train)
    logger.info('✅ Modelo treinado com sucesso!')

    # Avaliação do modelo
    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    logger.info(f'F1 score: {f1:.2f}')
    print(f'F1 score: {f1:.2f}')

    # Salvar o modelo treinado
    joblib.dump(model, model_output_filepath)
    logger.info(f'✅ Modelo salvo em {model_output_filepath}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())

    main()
