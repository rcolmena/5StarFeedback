# Script de Python que contiene la función principal que se llama desde la app de Flask para realizar el análisis de los
# datos una vez el usuario introduce el nombre de un restaurante.

# Importación de librerías y dependencias
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import random
import nltk
import re
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import heapq
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pymongo
import base64
from datetime import datetime


# 1) SCRAPER ===========================================================================================================

# TRIPADVISOR
def reviewScraperTripAdvisor(restaurant_name):
    '''
    Función para el scraping en TripAdvisor
    :param restaurant_name: nombre del restaurante introducido por el usuario
    :return: df: DataFrame con las reseñas scrapeadas
    '''


    # El origen del scraper será en este caso la página principal de restaurantes de TripAdvisor
    url_ta = 'https://www.tripadvisor.es/Restaurants'
    origin = 'TripAdvisor'

    # Número de páginas para obtener las reseñas
    num_page = 15

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")

    driver = webdriver.Chrome(options=options)
    driver.get(url_ta)

    # Espera hasta que se abra el botón de las cookies
    time.sleep(random.uniform(5.0, 7.0))

    # Click en el botón de las cookies
    driver.find_element(by=By.XPATH,
                        value='/html/body/div[10]/div[2]/div/div[2]/div[1]/div/div[2]/div/div[1]/button').click()

    time.sleep(random.uniform(1.0, 2.0))

    # Encuentra la barra de búsqueda
    item = driver.find_element(By.XPATH,
                               '/html/body/div[2]/div/div[1]/div/div/div/div/div/div[2]/div/div/div/form/input[1]')

    # Realiza la búsqueda con el nombre del restaurante
    item.send_keys(restaurant_name)

    time.sleep(random.uniform(2.0, 3.0))

    search = driver.find_element(By.XPATH,
                                 '/html/body/div[2]/div/div[1]/div/div/div/div/div/div[2]/div/div/div/form/div/a[1]')

    time.sleep(random.uniform(2.0, 3.0))

    # Se obtiene el link del restaurante en concreto tras realizar la búsqueda
    link = search.get_attribute('href')

    # Se accede a la página
    driver.get(link)

    # SCRAPER ---------------------------------------------------------------------------------------------
    contador = 1
    nextPage = True

    lst = []

    while nextPage and (contador < num_page):

        time.sleep(random.uniform(5.0, 7.0))  # Espera a que se carguen las reseñas

        reviews = driver.find_elements(by=By.XPATH, value="//div[@class='ui_column is-9']")
        num_page_items = min(len(reviews), 10)

        if num_page_items < 10:
            break

        # Bucle por las reseñas encontradas
        for i in range(num_page_items):

            # Para obtener la puntuación, la fecha y la reseña
            date = reviews[i].find_element(by=By.XPATH, value=".//span[@class='ratingDate']").get_attribute("title")
            review1 = reviews[i].find_element(by=By.XPATH, value=".//p[@class='partial_entry']").get_attribute("innerHTML")

            try:
                review2 = reviews[i].find_element(by=By.XPATH, value=".//span[@class='postSnippet']").text.replace("\n","")
            except:
                review2 = ""

            review = review1 + review2

            data = [date, review, origin, restaurant_name]

            lst.append(data)

        try:
            # Para la primera página
            nextButton = driver.find_element(by=By.XPATH,
                                             value='//*[@id="taplc_location_reviews_list_resp_rr_resp_0"]/div/div[13]/div/div/a[2]')
            nextButton.click()
            contador = contador + 1

        except:
            try:
                # Para el resto de páginas
                nextButton = driver.find_element(by=By.XPATH,
                                                 value='//*[@id="taplc_location_reviews_list_resp_rr_resp_0"]/div/div[12]/div/div/a[2]')
                nextButton.click()
                contador = contador + 1
            except:
                nextPage = False

    # Se cierra el navegador
    driver.close()

    df = pd.DataFrame(data=lst, columns=['Date', 'Review', 'Origin', 'Restaurant'])

    return df


# GOOGLE MAPS
def reviewScraperGoogle(restaurant_name):
    '''
    Función para el scraping en Google Maps
    :param restaurant_name: nombre del restaurante introducido por el usuario
    :return: df: DataFrame con las reseñas scrapeadas
    '''

    # El origen del scraper será en este caso la página principal de Google Maps
    url_g = 'https://www.google.es/maps/'
    origin = 'Google Maps'

    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)

    driver.get(url_g)
    time.sleep(random.uniform(5.0, 7.0))

    # Click en el botón de las cookies
    driver.find_element(by=By.XPATH,
                        value='//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[3]/div[1]/div[1]/form[2]/div/div/button').click()

    time.sleep(1)

    # Encuentra la barra de búsqueda
    item = driver.find_element(By.XPATH,
                               '/html/body/div[3]/div[9]/div[3]/div[1]/div[1]/div[1]/div[2]/form/div[2]/div[3]/div/input[1]')

    # Realiza la búsqueda con el nombre del restaurante
    item.send_keys(restaurant_name)

    # Click en la lupa para buscar el nombre del restaurante
    driver.find_element(By.XPATH, '//*[@id="searchbox-searchbutton"]').click()

    time.sleep(random.uniform(5.0, 7.0))

    # Click para ver todas las reseñas
    driver.find_element(By.XPATH,
                        '/html/body/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[3]/div/div/button[2]').click()

    time.sleep(random.uniform(5.0, 7.0))

    # Click en el botón de ordenar
    driver.find_element(By.XPATH,
                        '/html/body/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[3]/div[8]/div[2]/button').click()

    time.sleep(random.uniform(2.0, 3.0))

    # Cuatro categorías: Más útiles [1], Más recientes [2], Valoración más alta [3], Valoración más baja [4]
    driver.find_element(By.XPATH, '//*[@id="action-menu"]/div[1]').click()

    # SCROLL -----------------------------------------------------------------------------------------------------------

    # El scroll va a hacer que se recopilen más o menos reseñas

    SCROLL_PAUSE_TIME = 6

    # Altura del scroll
    last_height = driver.execute_script("return document.body.scrollHeight")

    number = 0

    while True:
        number = number + 1

        # Scroll hasta abajo
        ele = driver.find_element(By.XPATH, '/html/body/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[3]')

        # Modificando el número de scrollBy aumenta o disminuye el número de reseñas
        driver.execute_script('arguments[0].scrollBy(0, 10000);', ele)

        # Espera hasta que se cargue la página
        time.sleep(SCROLL_PAUSE_TIME)

        ele = driver.find_element(By.XPATH, '/html/body/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[3]')
        new_height = driver.execute_script("return arguments[0].scrollHeight", ele)

        if number == 5:
            break
        if new_height == last_height:
            break
        last_height = new_height

    # FIN DEL SCROLL ---------------------------------------------------------------------------------------------------

    # Botón de "Más" (solo para las reseñas largas)
    button = driver.find_elements(By.TAG_NAME, 'button')

    for m in button:
        if m.text == "Más":
            m.click()
    time.sleep(random.uniform(5.0, 7.0))

    date = driver.find_elements(By.CLASS_NAME, 'rsqaWe')
    review = driver.find_elements(By.CLASS_NAME, 'wiI7pd')

    lst = []
    for k, l in zip(date, review):
        date = k.text
        review = l.text

        data = [date, review, origin, restaurant_name]

        lst.append(data)

    df = pd.DataFrame(data=lst, columns=['Date', 'Review', 'Origin', 'Restaurant'])

    # Se cierra el navegador
    driver.close()

    return df


# 2) CLEANER AND TOKENIZER =============================================================================================

def cleaner_tokenizer(df):
    '''
    Función para la limpieza del texto
    :param df: DataFrame de las reseñas scrapeadas
    :return: df: DatFrame con las reseñas tokenizadas en frases y su correspondiente información
    '''

    # Se eliminan las reseñas que no tengan texto
    df = df.dropna().reset_index(drop=True)

    def clean_text(text):
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', text).replace('\n', '').replace('...', ' ').replace('Más', '')
        return text

    def remove_icons(text):
        clean = re.compile("["
            u"\U0001F600-\U0001F64F"  # icons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          "]+", re.UNICODE)

        text = re.sub(clean, '', text)
        return text

    for i in range(len(df)):
        df['Review'][i] = clean_text(df['Review'].iloc[i])
        df['Review'][i] = remove_icons(df['Review'].iloc[i])

    df['Review'] = df['Review'].apply(nltk.sent_tokenize)

    df = df.explode('Review')
    df = df.reset_index(drop=True)

    return df


# 3) CATEGORIZATION ====================================================================================================

def categorization(df):
    '''
    Función para la clasificación de las reseñas
    :param df: DataFrame con las reseñas categorizadas
    :return: df: mismo DataFrame añadiendo la columna Category, con la categoría de cada frase
    '''

    # Se carga el modelo de categorización
    tfidfFile = "back/tfidf.pkl"
    with open(tfidfFile, 'rb') as file:
        tfidf_vectorizador = pickle.load(file)

    classifierFile = "back/classifier.pkl"
    with open(classifierFile, 'rb') as file:
        modelo_final = pickle.load(file)

    # Se clasifican las reseñas
    df['Review'] = df['Review'].dropna().reset_index(drop=True)
    revs = tfidf_vectorizador.transform(df['Review'].values.astype('U'))
    predicciones_test = modelo_final.predict(X=revs)

    df['Category'] = predicciones_test
    df['Category'] = df['Category'].str.capitalize()

    return df


# 4) SENTIMENT ANALYSIS ================================================================================================

def sentiment(cat_df):
    '''
    Función para el análisis de sentimiento
    :param cat_df: DataFrame categorizado
    :return: cat_df: DataFrame categorizado añadiendo la columna con el sentimiento de cada frase
    '''

    # Se carga el modelo preentrenado de NLP Town
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    # Se aplica el modelo a las frases de las reseñas obtenidas
    def get_sentiment(review):
        tokens = tokenizer.encode(review, return_tensors='pt')
        result = model(tokens)
        return int(torch.argmax(result.logits)) + 1

    cat_df['Sentiment'] = cat_df['Review'].apply(get_sentiment)

    return cat_df


# 5) SUMMARIZING =======================================================================================================

def summarize(sent_df):
    '''
    Función para el resumen de las reseñas
    :param sent_df: DataFrame con el sentimiento de las frases
    :return: summary(str): resumen de las reseñas que entran a la función
    '''

    text = ' '.join(sent_df['Review'])

    # Limpieza de texto
    text = text.replace("", "")
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    formatted_text = re.sub('[^a-zA-Z]', ' ', text)
    formatted_text = re.sub(r'\s+', ' ', formatted_text)
    sentence_list = nltk.sent_tokenize(text)

    # Limpieza de stopwords
    stopwords = nltk.corpus.stopwords.words('spanish')

    word_frequencies = {}

    # Cálculo de la importancia de las palabras en función de su frecuencia
    for word in nltk.word_tokenize(formatted_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
        maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)
        sentence_scores = {}

    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    # Se selecciona la frase considerada más relevante
    summary_sentences = heapq.nlargest(1, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)

    return summary


# 6) WORDCLOUD =========================================================================================================

def wordcloud(df):
    '''
    Función para la creación de los gráficos wordcloud
    :param df: DataFrame con todas las frases
    :return: wc(str): cadena de texto equivalente a la imagen del wordcloud en base64
    '''

    # Se carga el modelo de Spacy para aplicar la técnica POS Tagging
    nlp = spacy.load('es_core_news_md')

    text = ' '.join(df['Review'])
    document = nlp(text)

    # Se extraen los adjetivos del texto
    adjs = []
    for token in document:
        if token.pos_ == 'ADJ':
            adjs.append(token.text)

    # Se carga el modelo de análisis de sentimiento para analizar los adjetivos
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    # Se aplica el modelo de análisis de sentimiento a los adjetivos encontrados
    def sentiment_score(review):
        tokens = tokenizer.encode(review, return_tensors='pt')
        result = model(tokens)
        return int(torch.argmax(result.logits)) + 1

    lst_adjs = []
    for adj in adjs:
        lst_adjs.append(sentiment_score(adj))

    df_adj = pd.DataFrame({'Adjetivo': adjs, 'Sentimiento': lst_adjs})

    text = ' '.join(df_adj['Adjetivo'])

    # Para aplciar el código de color al wordcloud en función del sentimiento obtenido
    class SimpleGroupedColorFunc(object):

        def __init__(self, color_to_words, default_color):
            self.word_to_color = {word: color
                                  for (color, words) in color_to_words.items()
                                  for word in words}

            self.default_color = default_color

        def __call__(self, word, **kwargs):
            return self.word_to_color.get(word, self.default_color)

    wc = WordCloud(width=500, height=500, background_color='white').generate(text.lower())

    color_to_words = {

        '#FF0D0D': df_adj[df_adj['Sentimiento'] == 1]['Adjetivo'],
        '#FF8E15': df_adj[df_adj['Sentimiento'] == 2]['Adjetivo'],
        '#FAB733': df_adj[df_adj['Sentimiento'] == 3]['Adjetivo'],
        '#ACB334': df_adj[df_adj['Sentimiento'] == 4]['Adjetivo'],
        '#69B34C': df_adj[df_adj['Sentimiento'] == 5]['Adjetivo'],

    }
    default_color = 'grey'

    grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)
    wc.recolor(color_func=grouped_color_func)

    plt.figure(figsize=(7, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")

    # Se guarda la imagen del wordcloud
    path = f"wordcloud_{(df['Category'].values[0]).lower()}.png"
    wc.to_file(path)

    # Se genera un string en base64 a partir de la imagen para poder almacenar los gráficos en la base de datos
    with open(f"wordcloud_{(df['Category'].values[0]).lower()}.png", "rb") as imageFile:
        wc = base64.b64encode(imageFile.read())

    return wc


# 7) RECOMMENDATIONS ===================================================================================================

def recommendations(punt, sent_df):
    '''
    Función para generar las recomendaciones en función de los resultados obtenidos
    :param punt: puntuaciones para todas las categorías
    :param sent_df: DataFrame con las frases con categorías y análisis de sentimiento
    :return: rec_lst([]): lista de todas las recomendaciones generadas
    '''

    rec_lst = []

    # Comida [5]
    if punt[punt['Category'] == 'Comida']['Sentiment'].values[0] >= 4:
        rec_lst.append('Los clientes están muy satisfechos con la comida.')
    elif (punt[punt['Category'] == 'Comida']['Sentiment'].values[0] < 4) & (punt[punt['Category'] == 'Comida']['Sentiment'].values[0] >= 3):
        rec_lst.append('Los clientes están satisfechos con la comida.')
    else:
        rec_lst.append('Los clientes no están satisfechos con la comida.')

    if (len(sent_df.loc[sent_df['Review'].str.contains("entrantes|entradas")]) != 0) &\
            (sent_df.loc[sent_df['Review'].str.contains("entrantes|entradas"), 'Sentiment'].mean() >= 3.5):
        rec_lst.append('No modificar los entrantes.')
    elif(len(sent_df.loc[sent_df['Review'].str.contains("entrantes|entradas")]) != 0) &\
            (sent_df.loc[sent_df['Review'].str.contains("entrantes|entradas"), 'Sentiment'].mean() < 3.5):
        rec_lst.append('Cambiar o mejorar los entrantes.')
    elif(len(sent_df.loc[sent_df['Review'].str.contains("entrantes|entradas")]) == 0):
        rec_lst.append('Los clientes no han opinado acerca de los entrantes.')

    if (len(sent_df.loc[sent_df['Review'].str.contains("carne|carnes")]) != 0) &\
            (sent_df.loc[sent_df['Review'].str.contains("carne|carnes"), 'Sentiment'].mean() >= 3.5):
        rec_lst.append('La carne le gusta a los clientes.')
    elif (len(sent_df.loc[sent_df['Review'].str.contains("carne|carnes")]) != 0) & \
            (sent_df.loc[sent_df['Review'].str.contains("carne|carnes"), 'Sentiment'].mean() < 3.5):
        rec_lst.append('Cambiar o mejorar la carne.')
    elif (len(sent_df.loc[sent_df['Review'].str.contains("carne|carnes")]) == 0):
        rec_lst.append('Los clientes no han opinado acerca de la carne')

    if (len(sent_df.loc[sent_df['Review'].str.contains("pescado|pescados")]) != 0) & \
            (sent_df.loc[sent_df['Review'].str.contains("pescado|pescados"), 'Sentiment'].mean() >= 3.5):
        rec_lst.append('El pescado le gusta a los clientes.')
    elif (len(sent_df.loc[sent_df['Review'].str.contains("pescado|pescados")]) != 0) & \
            (sent_df.loc[sent_df['Review'].str.contains("pescado|pescados"), 'Sentiment'].mean() < 3.5):
        rec_lst.append('Cambiar o mejorar el pescado.')
    elif (len(sent_df.loc[sent_df['Review'].str.contains("pescado")]) == 0):
        rec_lst.append('Los clientes no han opinado acerca de el pescado.')

    if (len(sent_df.loc[sent_df['Review'].str.contains("postre|postres")]) != 0) & \
            (sent_df.loc[sent_df['Review'].str.contains("postre|postres"), 'Sentiment'].mean() >= 3.5):
        rec_lst.append('No modificar los postres.')
    elif (len(sent_df.loc[sent_df['Review'].str.contains("postre|postres")]) != 0) & \
            (sent_df.loc[sent_df['Review'].str.contains("postre|postres"), 'Sentiment'].mean() < 3.5):
        rec_lst.append('Cambiar o mejorar los postres.')
    elif (len(sent_df.loc[sent_df['Review'].str.contains("postre|postres")]) == 0):
        rec_lst.append('Los clientes no han opinado acerca de los postres.')

    # Servicio [3]
    if len(sent_df.loc[sent_df['Review'].str.contains("desagradable")]) >= 5:
        rec_lst.append('El servicio está siendo desagradable para los clientes.')
    else:
        rec_lst.append('El servicio está siendo agradable para los clientes.')

    if len(sent_df.loc[sent_df['Review'].str.contains("lento")]) >= 5:
        rec_lst.append('El servicio es demasiado lento.')
    else:
        rec_lst.append('La velocidad del servicio es adecuada.')

    if len(sent_df.loc[sent_df['Review'].str.contains("desorganizado")]) >= 5:
        rec_lst.append('Mejorar la organización del servicio.')
    else:
        rec_lst.append('La organización del servicio es correcta.')

    # Precio [1]
    if len(sent_df.loc[sent_df['Review'].str.contains("caro|alto")]) >= 5:
        rec_lst.append('Reducir los precios.')
    else:
        rec_lst.append('La relación calidad-precio es correcta.')

    # Limpieza [1]
    if punt[punt['Category'] == 'Limpieza']['Sentiment'].values[0] >= 3.5:
        rec_lst.append('La limpieza es adecuada.')
    else:
        rec_lst.append('Mejorar la limpieza del establecimiento.')

    # Ambiente [2]
    if punt[punt['Category'] == 'Limpieza']['Sentiment'].values[0] >= 3.5:
        rec_lst.append('Los clientes están conformes con el ambiente.')
    else:
        rec_lst.append('A los clientes no les resulta agradable el ambiente.')

    if len(sent_df.loc[sent_df['Review'].str.contains("rudioso|ruido|bullicio")]) >= 0:
        rec_lst.append('Intentar disminuir el ruido.')
    else:
        rec_lst.append('La cantidad de ruido es adecuada.')

    return rec_lst


# 8) DB INSERTION ======================================================================================================

def db_insertion(restaurant_name, sent_df, sum_pos, sum_neg, wc_df, punt_general, punt, rec_lst, MONGODB_URI):
    '''
    Función para la inserción en la base de datos
    :param restaurant_name: nombre del restaurante
    :param sent_df: DataFrame con las frases con categorías y análisis de sentimiento
    :param sum_pos: resumen de reseñas positivas
    :param sum_neg: resumen de reseñas negativas
    :param wc_df([]): strings correspondientes a los wordclouds
    :param punt_general(int): puntuación media
    :param punt: DataFrame con las puntuaciones por categoría
    :param rec_lst: recomendaciones
    :return: -
    '''

    # Conexión a la base de datos
    con = pymongo.MongoClient(MONGODB_URI)

    # Selección de la base de datos
    db = con.FiveStarFeedback

    # Selección de la colección
    restaurant = db.Restaurants

    restaurant_name = restaurant_name.lower().replace(' ', '_')
    df = [sent_df[['Date', 'Review', 'Origin', 'Category', 'Sentiment']].to_dict('records')]
    df2 = [wc_df.to_dict('records')]
    df3 = [punt.to_dict('records')]
    df4 = [pd.DataFrame({'Recomendación': rec_lst}).to_dict('records')]

    date = datetime.today().strftime('%d-%m-%Y')

    d = {'nombre_restaurante': restaurant_name, 'fecha': date, 'resumen_positivo': sum_pos, 'resumen_negativo': sum_neg,
         'sentimiento': punt_general, 'sentimiento_categoria': df3, 'reseñas': df, 'wordclouds': df2,
         'recomendaciones': df4}

    df = pd.DataFrame(data=d, index=[0])

    # Inserción de los datos
    restaurant.insert_many(df.to_dict('records'))


# MAIN =================================================================================================================

def data_processing(restaurant_name, MONGODB_URI):
    '''
    Función principal de la aplicación. Llama a todas las funciones de este mismo Script.
    :param restaurant_name: nombre del restaurante
    :param MONGODB_URI: connection string para la base de datos
    :return: -
    '''

    # SCRAPER
    try:
        reviews_t_df = reviewScraperTripAdvisor(restaurant_name)
    except:
        # Si no se encuentran resultados en TripAdvisor, se devuelve un DataFrame vacío
        reviews_t_df = pd.DataFrame(columns=['Date', 'Review', 'Origin', 'Restaurant'])

    try:
        reviews_g_df = reviewScraperGoogle(restaurant_name)
    except:
        # Si no se encuentran resultados en Google Maps, se devuelve un DataFrame vacío
        reviews_g_df = pd.DataFrame(columns=['Date', 'Review', 'Origin', 'Restaurant'])

    # CLEANER AND TOKENIZER
    clean_t = cleaner_tokenizer(reviews_t_df)
    clean_g = cleaner_tokenizer(reviews_g_df)
    clean_df = pd.concat([clean_t, clean_g]).reset_index(drop=True)

    if len(clean_df) != 0:

        # CLASSIFIER
        cat_df = categorization(clean_df)

        # SENTIMENT ANALYSIS
        sent_df = sentiment(cat_df)

        # SENTIMENT BY CATEGORY
        punt = round(sent_df.groupby('Category')['Sentiment'].mean(), 1)
        punt = punt.reset_index()
        # En caso de que no se encuentre alguna de las categorías, se asigna una puntuación neutra de 3
        if ('Comida' not in sent_df['Category'].values):
            punt = pd.concat([punt, (pd.DataFrame({'Category': 'Comida', 'Sentiment': 3.0}, index=[0]))])
        if ('Servicio' not in sent_df['Category'].values):
            punt = pd.concat([punt, (pd.DataFrame({'Category': 'Servicio', 'Sentiment': 3.0}, index=[0]))])
        if ('Precio' not in sent_df['Category'].values):
            punt = pd.concat([punt, (pd.DataFrame({'Category': 'Precio', 'Sentiment': 3.0}, index=[0]))])
        if ('Limpieza' not in sent_df['Category'].values):
            punt = pd.concat([punt, (pd.DataFrame({'Category': 'Limpieza', 'Sentiment': 3.0}, index=[0]))])
        if ('Ambiente' not in sent_df['Category'].values):
            punt = pd.concat([punt, (pd.DataFrame({'Category': 'Ambiente', 'Sentiment': 3.0}, index=[0]))])

        punt = punt.reset_index().sort_values(by='Category')
        punt_general = round(punt['Sentiment'].mean(), 1)

        # SUMMARIZER
        # Resumen positivo
        pos_df = sent_df[sent_df['Sentiment'].isin([4, 5])]
        # Resumen negativo
        neg_df = sent_df[sent_df['Sentiment'].isin([1, 2])]
        sum_pos = summarize(pos_df)
        sum_neg = summarize(neg_df)

        # WORDCLOUD
        cat = sent_df.groupby('Category').filter(lambda x: len(x) > 3)
        cat = cat.groupby('Category')
        wc_df = cat.apply(wordcloud).reset_index()
        wc_df.columns = ['Category', 'Plot']

        # RECOMMENDATIONS
        rec_lst = recommendations(punt, sent_df)

        # DB INSERTION
        db_insertion(restaurant_name, sent_df, sum_pos, sum_neg, wc_df, punt_general, punt, rec_lst, MONGODB_URI)

    else:
        print('No se han obtenido datos del restaurante introducido.')