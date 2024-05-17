# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import boto3

@click.command()
@click.argument('s3_bucket')
@click.argument('s3_object')
@click.argument('output_filepath', type=click.Path())
def main(s3_bucket, s3_object, output_filepath):
    """ Get data from S3 bucket and object and save it to a local file
    """
    logger = logging.getLogger(__name__)
    logger.info('⌛ Getting dataset from s3.')

    s3 = boto3.client('s3')
    s3.download_file(s3_bucket, s3_object, output_filepath)

    logger.info('✅ Dataset downloaded from s3.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
