# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd 
from src.data import _normalize_text as nt 

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('⌛ Iniciando processo de limpeza dos dados...')

    df = pd.read_csv(input_filepath, delimiter=';') 
    df_transformed = df.copy()
    logger.info('✅ Arquivo de dados carregado com sucesso!')

    df_transformed.drop("id_reclamacao", axis=1, inplace=True)
    df_transformed.drop("data_abertura", axis=1, inplace=True)

    df_transformed['descricao_reclamacao'] = df_transformed['descricao_reclamacao'].apply(nt.limpar_padroes)
    logger.info('✅ Ruídos removidos com sucesso!')

    df_transformed['descricao_reclamacao'] = df_transformed['descricao_reclamacao'].apply(nt.remove_punctuation)
    logger.info('✅ Pontuações removidas com sucesso!')

    #df_transformed['text'] = df_transformed['text'].apply(nt.remover_stopwords)
    #logger.info('✅ Textos limpos com sucesso!')

    logger.info('⌛ Conversão para CSV.')
    df_transformed.to_csv(output_filepath, index=False)
    logger.info('✅ Processo concluído.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())

    main()
