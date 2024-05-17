# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data import _normalize_text as nt

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs feature extraction scripts to turn cleaned data from (../processed) into
        feature matrix ready for modeling (saved in ../features).
    """
    logger = logging.getLogger(__name__)
    logger.info('⌛ Iniciando extração de features...')

    # Carregar o modelo de linguagem do SpaCy
    nlp = spacy.load('pt_core_news_sm')
    
    # Baixar os recursos necessários do NLTK
    nltk.download('stopwords')
    
    # Carregar o arquivo de dados
    try:
        df = pd.read_csv(input_filepath)
        logger.info('✅ Arquivo de dados carregado com sucesso!')
    except pd.errors.ParserError as e:
        logger.error(f'❌ Erro ao ler o arquivo CSV: {e}')
        return

    # Amostrar 10% dos dados
    df_sample = df.sample(frac=0.1, random_state=42)
    logger.info(f'✅ Amostra de 10% dos dados selecionada: {df_sample.shape[0]} linhas.')

    # Combinar stopwords do SpaCy e NLTK
    stops = list(set(nlp.Defaults.stop_words).union(set(nltk.corpus.stopwords.words('portuguese'))))

    # Vetorização usando TfidfVectorizer
    vect = TfidfVectorizer(ngram_range=(1,1), use_idf=True, stop_words=stops)
    vect.fit(df_sample['descricao_reclamacao'])
    text_vect = vect.transform(df_sample['descricao_reclamacao'])

    # Converter a matriz esparsa para um DataFrame
    feature_names = vect.get_feature_names_out()
    df_features = pd.DataFrame(text_vect.toarray(), columns=feature_names)

    # Adicionar a coluna de categorias ao DataFrame de features
    df_features['categoria'] = df_sample['categoria']

    # Verificar o tamanho do DataFrame
    logger.info(f'✅ DataFrame de features criado com {df_features.shape[0]} linhas e {df_features.shape[1]} colunas.')

    # Salvar os vetores em um arquivo CSV
    logger.info('⌛ Salvando as features no arquivo de saída...')
    try:
        df_features.to_csv(output_filepath, index=False)
        logger.info('✅ Extração de features concluída.')
    except Exception as e:
        logger.error(f'❌ Erro ao salvar o arquivo CSV: {e}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())

    main()
