import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib
from scipy import sparse
from dvclive import Live

@click.command()
@click.argument('input_filepath_features', type=click.Path(exists=True))
@click.argument('input_filepath_categories', type=click.Path(exists=True))
@click.argument('model_output_filepath', type=click.Path())
def main(input_filepath_features, input_filepath_categories, model_output_filepath):
    """ Runs training and evaluation scripts to train a model on the processed data and
        evaluate its performance.
    """
    with Live() as live:
        logger = logging.getLogger(__name__)
        logger.info('⌛ Iniciando o treinamento do modelo...')

        # Carregar a matriz esparsa de features
        try:
            x = sparse.load_npz(input_filepath_features)
            logger.info('✅ Matriz de features carregada com sucesso!')
        except Exception as e:
            logger.error(f'❌ Erro ao ler a matriz de features: {e}')
            return

        # Carregar a coluna de categorias
        try:
            y = pd.read_csv(input_filepath_categories)['categoria']
            logger.info('✅ Coluna de categorias carregada com sucesso!')
        except pd.errors.ParserError as e:
            logger.error(f'❌ Erro ao ler o arquivo CSV de categorias: {e}')
            return

        # Remover linhas com valores NaN
        mask = ~y.isna()
        x = x[mask]
        y = y[mask]
        logger.info(f'✅ DataFrame sem NaNs: {x.shape[0]} linhas restantes.')

        # Dividir os dados em conjuntos de treino e teste
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

        # Treinamento do modelo
        model = LogisticRegression(random_state=42)
        model.fit(x_train, y_train)
        logger.info('✅ Modelo treinado com sucesso!')

        # Avaliação do modelo
        y_pred = model.predict(x_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        logger.info(f'F1 score: {f1:.2f}')
        
        # Logar a métrica com DVCLive
        live.log_metric('F1 Score', f1)

        # Salvar o modelo treinado
        joblib.dump(model, model_output_filepath)
        logger.info(f'✅ Modelo salvo em {model_output_filepath}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())

    main()

