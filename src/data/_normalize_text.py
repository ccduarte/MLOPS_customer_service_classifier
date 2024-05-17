import pandas as pd
import string
import re
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def limpar_padroes(texto):
    """
    Função criada com o objetivo de retirar padrões que não são úteis para a classificação
    """
    padroes = [
        'XXXX', 'XXX', 'xxxx', 'Xxxx', 'xx', 'XX',
        r'\r', r'\n',
        r'\d{1,2}/\d{1,2}/\d{4}',  # Data xx/xx/xxxx
        r'\d{2}/\d{2}/\d{4}',  # Data xx/xx/xxxx (sem espaço)
        r'\d{1,2}/\d{1,2}/\d{4}/\d{2}/\d{2}',  # Data xx/xx/xxxx/xx/xx
        r'/Xxxx', '\(', '\)', "'", '"',
        r'\{\$[\d\.,]+\}',
        r'XX/XX/',  # Padrão XX/XX/
        r'xx/xx/',  # Padrão xx/xx/
        r'//',  # Padrão //
        r'/xx/xx '  # Padrão /xx/xx com espaço
    ]

    for padrao in padroes:
        texto = re.sub(padrao, '', texto, flags=re.IGNORECASE)

    # Substituir "{$ qualquer_valor}" por "R$ qualquer_valor"
    texto = re.sub(r'\{\$ *([\d\.,]+)\}', lambda match: '$ ' + match.group(1), texto)

    return texto
def remove_punctuation(text):
    """
    Função que remove pontuação
    """
    table = str.maketrans('', '', string.punctuation)
    text = text.translate(table)
    return text

def remover_stopwords(texto, metodo='ambos'):
    """
    Função para remover stopwords de um texto.

    Parâmetros:
        texto (str): O texto a ser processado.
        metodo (str): Método de remoção de stopwords a ser utilizado. Pode ser 'spacy', 'nltk' ou 'ambos'.

    Retorna:
        str: O texto sem as stopwords.
    """

    nltk.download('stopwords')
    nlp = spacy.load('pt_core_news_sm')

    if metodo == 'spacy':
        nlp = spacy.load('pt_core_news_sm')
        stops = nlp.Defaults.stop_words
    elif metodo == 'nltk':
        stops = nltk.corpus.stopwords.words('portuguese')
    elif metodo == 'ambos':
        stops = list(set(nlp.Defaults.stop_words).union(set(nltk.corpus.stopwords.words('portuguese'))))
    else:
        raise ValueError("Método de remoção de stopwords inválido. Use 'spacy', 'nltk' ou 'ambos'.")
    return stops

def vetorizar_texto(df, tipo='BoW'):
    """
    Função para vetorizar o texto do DataFrame usando diferentes tipos de contagem de termos.

    Parâmetros:
        df (DataFrame): DataFrame contendo a coluna 'descricao_reclamacao' com os textos a serem vetorizados.
        tipo (str): Tipo de contagem de termos a ser aplicado. Pode ser 'BoW', 'Bigrama', 'Trigrama', 'TF' ou 'TF-IDF'.

    Retorna:
        sparse matrix: Matriz de características (X_train) resultante da vetorização.
    """
    if tipo == 'BoW':
        vect = CountVectorizer(ngram_range=(1,1))
    elif tipo == 'Bigrama':
        vect = CountVectorizer(ngram_range=(2,2))
    elif tipo == 'Trigrama':
        vect = CountVectorizer(ngram_range=(3,3))
    elif tipo == 'TF':
        vect = TfidfVectorizer(ngram_range=(1,1), use_idf=False, norm='l1')
    elif tipo == 'TF-IDF':
        vect = TfidfVectorizer(ngram_range=(1,1), use_idf=True)
    else:
        raise ValueError("Tipo de contagem de termos inválido. Use 'BoW', 'Bigrama', 'Trigrama', 'TF' ou 'TF-IDF'.")

    vect.fit(df['descricao_reclamacao'])
    X_train = vect.transform(df['descricao_reclamacao'])

    return X_train
