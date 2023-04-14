# Script de Python para realizar un análisis entre las distintas alternativas para el sentiment analysis
# Se compara la construcción a mano de un modelo SVM y el uso del transformer BERT BASE MULTILINGUAL

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# SENTIMENT CLASSIFIER (MODELS) =========================================================================

df = pd.read_csv('reviews_classifier_sentiment.csv', header=0, index_col=0, delimiter=';')
df['Sentiment'] = df['Sentiment'].astype(str)
df = df.reset_index(drop=True)

# División de los datos en entrenamiento y test
datos_X = df.loc[df.Sentiment.isin(['1', '2', '3', '4', '5']), 'Review']
datos_y = df.loc[df.Sentiment.isin(['1', '2', '3', '4', '5']), 'Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    datos_X,
    datos_y,
    test_size=0.2,
    random_state=123

)

def limpiar_tokenizar(texto):
    # Se convierte el texto a minúsculas
    nuevo_texto = texto.lower()
    # Eliminación de páginas web (palabras que empiezan por "http")
    nuevo_texto = re.sub('http\S+', ' ', nuevo_texto)
    # Eliminación de signos de puntuación
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    nuevo_texto = re.sub(regex, ' ', nuevo_texto)
    # Eliminación de números
    nuevo_texto = re.sub("\d+", ' ', nuevo_texto)
    # Eliminación de espacios en blanco múltiples
    nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
    # Tokenización por palabras individuales
    nuevo_texto = nuevo_texto.split(sep=' ')
    # Eliminación de tokens con una longitud < 2
    nuevo_texto = [token for token in nuevo_texto if len(token) > 1]

    return (nuevo_texto)

stop_words = list(stopwords.words('spanish'))

# Matriz TF-IDF
tfidf_vectorizador = TfidfVectorizer(
                        tokenizer  = limpiar_tokenizar,
                        min_df     = 3,
                        stop_words = stop_words
                    )
tfidf_vectorizador.fit(X_train)

tfidf_train = tfidf_vectorizador.transform(X_train)
tfidf_test  = tfidf_vectorizador.transform(X_test)

# Entrenamiento del modelo SVM
modelo_svm_lineal = svm.SVC(kernel = "linear", C = 1.0)
modelo_svm_lineal.fit(X = tfidf_train, y = y_train)

# Construcción del grid
param_grid = {'C': np.logspace(-5, 3, 10)}

# Validación cruzada
grid = GridSearchCV(
        estimator  = svm.SVC(kernel= "linear"),
        param_grid = param_grid,
        scoring    = 'accuracy',
        n_jobs     = -1,
        cv         = 5, 
        verbose    = 0,
        return_train_score = True
      )

_ = grid.fit(X = tfidf_train, y = y_train)

# Resultados del grid
resultados = pd.DataFrame(grid.cv_results_)
resultados.filter(regex = '(param.*|mean_t|std_t)')\
    .drop(columns = 'params')\
    .sort_values('mean_test_score', ascending = False)

# Mejores hiperparámetros encontrados por validación cruzada
print("Mejores hiperparámetros encontrados (cv):")
print(grid.best_params_, ":", grid.best_score_, grid.scoring)

modelo_final = grid.best_estimator_

# Evaluación
predicciones_test = modelo_final.predict(X = tfidf_test)

print("Error de test")
print(f"Número de clasificaciones erróneas de un total de {tfidf_test.shape[0]} " \
      f"clasificaciones: {(y_test != predicciones_test).sum()}")
print(f"% de error: {100*(y_test != predicciones_test).mean()}")

print("Matriz de confusión")
print("-------------------")
print(pd.DataFrame(confusion_matrix(y_true=y_test, y_pred=predicciones_test),
             columns=['1', '2', '3', '4', '5'],
             index=['1', '2', '3', '4', '5']))

tfidfFile = "tfidf.pkl"
with open(tfidfFile, 'wb') as file:
    pickle.dump(tfidf_vectorizador, file)

classifierFile = "classifier.pkl"
with open(classifierFile, 'wb') as file:
    pickle.dump(modelo_final, file)

# COMPARISON BETWEEN SVM AND BERT =========================================================================

df['Review'] = df['Review'].dropna().reset_index(drop=True)
revs = tfidf_vectorizador.transform(df['Review'].values.astype('U'))
predicciones_test = modelo_final.predict(X=revs)

df['Sentiment SVM'] = predicciones_test

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

lst = []
for phrase in df['Review']:
    lst.append(sentiment_score(phrase))

df['Sentiment BERT'] = lst
df['Sentiment BERT'] = df['Sentiment BERT'].astype(str)

print(df[df['Sentiment BERT']==df['Sentiment SVM']])

print('Precisión por una estrella para SVM')
print(df[(df['Sentiment']==df['Sentiment SVM']+1) | (df['Sentiment']==df['Sentiment SVM']-1) | (df['Sentiment']==df['Sentiment SVM'])])
