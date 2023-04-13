from flask import Flask, render_template, request
from pymongo import MongoClient
import pandas as pd
import base64
from datetime import datetime, timedelta
import re
from back.main import data_processing
import os

import warnings
warnings.filterwarnings("ignore")

MONGODB_URI = os.getenv('MONGODB_URI')

con = MongoClient(MONGODB_URI)
db = con.FiveStarFeedback
restaurant = db.Restaurants

app = Flask(__name__)


@app.route('/')
def busqueda():
    return render_template("busqueda.html")


@app.route('/results', methods=['GET'])
def mostrarResultados():

    nombre_restaurante = request.values.get('nombre_restaurante')

    query = {'nombre_restaurante': nombre_restaurante}
    cursor = restaurant.find(query)
    data = pd.DataFrame(list(cursor))

    if len(data) == 0:
        data_processing(nombre_restaurante)
    else:
        fecha = data['fecha'][0]
        fecha = datetime.strptime(fecha, '%d-%m-%Y')
        if fecha < (datetime.today() - timedelta(days=90)):
            data_processing(nombre_restaurante)

    query = {'nombre_restaurante': nombre_restaurante}
    cursor = restaurant.find(query)
    data = pd.DataFrame(list(cursor))

    if len(data) == 0:
        return render_template("noresults.html", nombre_restaurante=nombre_restaurante)
    else:
        # DB Data
        nombre_restaurante = data['nombre_restaurante'][0]
        resumen_positivo = data['resumen_positivo'][0]
        resumen_negativo = data['resumen_negativo'][0]
        puntuacion = data['sentimiento'][0]
        punt_ambiente = data['sentimiento_categoria'][0][0]['Sentiment']
        punt_comida = data['sentimiento_categoria'][0][1]['Sentiment']
        punt_limpieza = data['sentimiento_categoria'][0][2]['Sentiment']
        punt_precio = data['sentimiento_categoria'][0][3]['Sentiment']
        punt_servicio = data['sentimiento_categoria'][0][4]['Sentiment']

        wc = pd.DataFrame(data['wordclouds'][0])

        try:
            wc_comida = wc[wc['Category'] == 'Comida']['Plot'].values[0].decode('utf-8')
        except:
            with open("static/images/nwc_comida.png", "rb") as imageFile:
                wc_comida = base64.b64encode(imageFile.read()).decode('utf-8')

        try:
            wc_precio = wc[wc['Category'] == 'Precio']['Plot'].values[0].decode('utf-8')
        except:
            with open("static/images/nwc_precio.png", "rb") as imageFile:
                wc_precio = base64.b64encode(imageFile.read()).decode('utf-8')

        try:
            wc_servicio = wc[wc['Category'] == 'Servicio']['Plot'].values[0].decode('utf-8')
        except:
            with open("static/images/nwc_servicio.png", "rb") as imageFile:
                wc_servicio = base64.b64encode(imageFile.read()).decode('utf-8')

        try:
            wc_limpieza = wc[wc['Category'] == 'Limpieza']['Plot'].values[0].decode('utf-8')
        except:
            with open("static/images/nwc_limpieza.png", "rb") as imageFile:
                wc_limpieza = base64.b64encode(imageFile.read()).decode('utf-8')

        try:
            wc_ambiente = wc[wc['Category'] == 'Ambiente']['Plot'].values[0].decode('utf-8')
        except:
            with open("static/images/nwc_ambiente.png", "rb") as imageFile:
                wc_ambiente = base64.b64encode(imageFile.read()).decode('utf-8')

        rec0 = data['recomendaciones'][0][0]['Recomendación']
        rec1 = data['recomendaciones'][0][1]['Recomendación']
        rec2 = data['recomendaciones'][0][2]['Recomendación']
        rec3 = data['recomendaciones'][0][3]['Recomendación']
        rec4 = data['recomendaciones'][0][4]['Recomendación']
        rec5 = data['recomendaciones'][0][5]['Recomendación']
        rec6 = data['recomendaciones'][0][6]['Recomendación']
        rec7 = data['recomendaciones'][0][7]['Recomendación']
        rec8 = data['recomendaciones'][0][8]['Recomendación']
        rec9 = data['recomendaciones'][0][9]['Recomendación']
        rec10 = data['recomendaciones'][0][10]['Recomendación']
        rec11 = data['recomendaciones'][0][11]['Recomendación']

        return render_template("index.html", nombre_restaurante=nombre_restaurante, resumen_positivo=resumen_positivo,
                               resumen_negativo=resumen_negativo, puntuacion=puntuacion, punt_ambiente=punt_ambiente,
                               punt_comida=punt_comida, punt_limpieza=punt_limpieza, punt_precio=punt_precio,
                               punt_servicio=punt_servicio, wc_comida=wc_comida, wc_precio=wc_precio,
                               wc_servicio=wc_servicio, wc_limpieza=wc_limpieza, wc_ambiente=wc_ambiente, rec0=rec0,
                               rec1=rec1, rec2=rec2, rec3=rec3, rec4=rec4, rec5=rec5, rec6=rec6, rec7=rec7, rec8=rec8,
                               rec9=rec9, rec10=rec10, rec11=rec11)


if __name__ == '__main__':

    # Función de limpieza para el texto
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

        return nuevo_texto

    app.run()