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

# CATEGORY CLASSIFIER (MODELS) =========================================================================

df = pd.read_csv('reviews_classifier.csv', header=0, index_col=0)
df = df.reset_index(drop=True)

# División de los datos en entrenamiento y test
datos_X = df.loc[df.Category.isin(['comida', 'precio', 'servicio', 'ambiente', 'limpieza']), 'Review']
datos_y = df.loc[df.Category.isin(['comida', 'precio', 'servicio', 'ambiente', 'limpieza']), 'Category']

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
             columns = ['comida', 'precio', 'servicio', 'ambiente', 'limpieza'],
             index = ['comida', 'precio', 'servicio', 'ambiente', 'limpieza']))

tfidfFile = "tfidf.pkl"
with open(tfidfFile, 'wb') as file:
    pickle.dump(tfidf_vectorizador, file)

classifierFile = "classifier.pkl"
with open(classifierFile, 'wb') as file:
    pickle.dump(modelo_final, file)