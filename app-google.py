from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import re
import random
import time
from datetime import datetime
from googleapiclient.discovery import build
import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.errors import HttpError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from spellchecker import SpellChecker
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import spacy
import concurrent.futures
from pysentimiento import create_analyzer
import emoji

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# --- Environment variables ---
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") # Eliminado
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") # Eliminado
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Añadido

# --- Google Generative AI API Configuration ---
# Puedes cambiar 'gemini-1.5-flash-latest' si necesitas otro modelo
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
# GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")
GOOGLE_API_URL_TEMPLATE = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')

# Cargar modelo de spaCy para español
try:
    nlp = spacy.load("es_core_news_sm")
except:
    print("Modelo de spaCy no encontrado. Por favor, ejecuta: python -m spacy download es_core_news_sm")

# Diccionario de correcciones comunes
CORRECCIONES_COMUNES = {
    "ke": "que",
    "q": "que",
    "xq": "porque",
    "pq": "porque",
    "tb": "también",
    "tmb": "también",
    "tambn": "también",
    "x": "por",
    "d": "de",
    "k": "que",
    "q": "que",
    "w": "con",
    "c": "con",
    "m": "me",
    "t": "te",
    "s": "es",
    "xfa": "por favor",
    "pls": "por favor",
    "plz": "por favor",
    "thx": "gracias",
    "ty": "gracias",
    "np": "no problem",
    "yw": "de nada",
    "nw": "de nada",
    "np": "no hay problema",
    "nvm": "no importa",
    "idk": "no sé",
    "idc": "no me importa",
    "tbh": "para ser honesto",
    "imo": "en mi opinión",
    "imho": "en mi humilde opinión",
    "afaik": "por lo que sé",
    "afaict": "por lo que puedo ver",
    "afaics": "por lo que puedo ver",
    "afaict": "por lo que puedo ver",
    "afaics": "por lo que puedo ver",
    "afaict": "por lo que puedo ver",
    "afaics": "por lo que puedo ver",
    "afaict": "por lo que puedo ver",
    "afaics": "por lo que puedo ver",
}

# Al inicio del archivo (después de imports)
analyzer_sentimiento = create_analyzer(task="sentiment", lang="es")


def procesar_transcripcion(texto):
    """
    Procesa y limpia una transcripción de video.
    """
    try:
        # 1. Limpieza básica
        # Eliminar timestamps y marcas de subtítulos
        texto = re.sub(r'\[\d{2}:\d{2}:\d{2}\]|\[\d{2}:\d{2}\]', '', texto)
        texto = re.sub(r'\[.*?\]', '', texto)  # Eliminar [Música], [Aplausos], etc.

        # Eliminar caracteres especiales y emojis
        texto = re.sub(r'[^\w\s]', ' ', texto)

        # Convertir a minúsculas
        texto = texto.lower()

        # 2. Corrección de errores comunes
        # Aplicar diccionario de correcciones
        palabras = texto.split()
        palabras_corregidas = [CORRECCIONES_COMUNES.get(palabra, palabra) for palabra in palabras]
        texto = ' '.join(palabras_corregidas)

        # Usar SpellChecker para correcciones adicionales
        spell = SpellChecker(language='es')
        palabras = texto.split()
        palabras_corregidas = []
        for palabra in palabras:
            if len(palabra) > 2:  # Ignorar palabras muy cortas
                correccion = spell.correction(palabra)
                palabras_corregidas.append(correccion if correccion else palabra)
            else:
                palabras_corregidas.append(palabra)
        texto = ' '.join(palabras_corregidas)

        # 3. Eliminar stopwords
        stop_words = set(stopwords.words('spanish'))
        palabras = word_tokenize(texto)
        palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words]
        texto = ' '.join(palabras_filtradas)

        # 4. Eliminar redundancias
        # Usar TextBlob para detectar y eliminar repeticiones
        blob = TextBlob(texto)
        oraciones = blob.sentences
        oraciones_unicas = []
        for oracion in oraciones:
            if oracion not in oraciones_unicas:
                oraciones_unicas.append(str(oracion))
        texto = ' '.join(oraciones_unicas)

        # 5. Resumen del texto
        # Usar TextRank para extraer las oraciones más importantes
        parser = PlaintextParser.from_string(texto, Tokenizer("spanish"))
        summarizer = TextRankSummarizer()
        resumen = summarizer(parser.document, sentences_count=5)  # Obtener 5 oraciones más importantes
        texto = ' '.join([str(sentence) for sentence in resumen])

        # 6. Análisis de relevancia por segmentos
        # Dividir en segmentos de 30 segundos (aproximadamente)
        segmentos = sent_tokenize(texto)
        segmentos_relevantes = []

        # Calcular TF-IDF para cada segmento
        vectorizer = TfidfVectorizer(stop_words='spanish')
        try:
            tfidf_matrix = vectorizer.fit_transform(segmentos)
            # Calcular la densidad de palabras clave para cada segmento
            densidades = np.array(tfidf_matrix.sum(axis=1)).flatten()
            # Seleccionar segmentos con alta densidad
            umbral = np.mean(densidades)
            for i, densidad in enumerate(densidades):
                if densidad > umbral:
                    segmentos_relevantes.append(segmentos[i])
        except:
            # Si hay error en el cálculo de TF-IDF, usar todos los segmentos
            segmentos_relevantes = segmentos

        texto = ' '.join(segmentos_relevantes)

        return texto

    except Exception as e:
        print(f"Error procesando transcripción: {e}")
        return texto  # Devolver texto original si hay error


def get_video_transcript(video_id, max_minutes=None):
    """Obtiene la transcripción de un video de YouTube en español"""
    try:
        # Intentar obtener la transcripción en español
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['es'])

        if max_minutes is None:
            # Procesar la transcripción completa
            texto_completo = " ".join([entry['text'] for entry in transcript])
            return procesar_transcripcion(texto_completo)

        max_seconds = max_minutes * 60
        text_parts = []
        current_time = 0

        for entry in transcript:
            if entry['start'] >= max_seconds:
                break
            text_parts.append(entry['text'])

        # Procesar la transcripción parcial
        texto_parcial = " ".join(text_parts)
        return procesar_transcripcion(texto_parcial)

    except Exception as e:
        if "No transcripts were found" in str(e) or "TranscriptsDisabled" in str(e):
            print(f"No se encontró transcripción en español o están deshabilitadas para el video {video_id}")
            # ✅ CAMBIO: En lugar de retornar None, retornar texto vacío
            return ""  # Cambiado de None a ""
        print(f"Error obteniendo transcripción para {video_id}: {e}")
        # ✅ CAMBIO: En lugar de retornar None, retornar texto vacío
        return ""  # Cambiado de None a ""


# --- Helper Function for Google Generative AI API Call (Minimal version) ---
def call_google_generative_api_for_text(prompt_text):
    """
    Calls the Google Generative AI API and returns the generated text.
    Minimal error handling for direct replacement.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    url = f"{GOOGLE_API_URL_TEMPLATE}?key={GOOGLE_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    # Estructura simple de la solicitud para Gemini
    body = json.dumps({
        "contents": [{
            "parts": [{"text": prompt_text}]
        }],
        # Mantener configuraciones de generación simples o excluirlas
        # "generationConfig": {
        #    "temperature": 0.7, # Coincide con el original
        # }
    })

    try:
        response = requests.post(url, headers=headers, data=body, timeout=120)
        response.raise_for_status()
        response_data = response.json()

        # Extracción estándar del texto generado por Gemini
        generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
        return generated_text

    except requests.exceptions.RequestException as req_err:
        print(f"Error en la solicitud a Google API: {req_err}")
        # Intenta mostrar la respuesta si existe
        try:
            print("Error Body:", response.text)
        except:
            pass
        raise  # Relanzar el error para que sea capturado por el llamador original
    except (KeyError, IndexError) as parse_err:
        print(f"Error parseando respuesta de Google API: {parse_err}")
        print("Raw Response Data:", response_data)
        raise ValueError(f"Formato de respuesta inesperado de Google API: {parse_err}")
    except Exception as e:
        print(f"Error inesperado en llamada a Google API: {e}")
        raise


def get_course_outline(prompt):
    """Genera el esquema del curso basado en el prompt y el nivel (Usando Google API)"""
    # Definir los niveles disponibles y sus características (sin cambios)
    levels = {
        "principiante": {
            "keywords": ["principiante", "inicial", "introductorio", "desde cero", "básico"],
            "num_sections": 4,
            "description": "Enfocado en conceptos fundamentales y primeros pasos",
            "depth": "superficial",
            "focus": "comprensión de conceptos básicos"
        },
        "intermedio": {
            "keywords": ["intermedio", "medio"],
            "num_sections": 6,
            "description": "Profundización en conceptos y aplicaciones prácticas",
            "depth": "moderada",
            "focus": "aplicación de conceptos y desarrollo de habilidades"
        },
        "avanzado": {
            "keywords": ["avanzado", "experto", "profesional"],
            "num_sections": 8,
            "description": "Temas especializados y técnicas avanzadas",
            "depth": "profunda",
            "focus": "optimización y casos de uso complejos"
        },
        "maestro": {
            "keywords": ["maestro", "master", "completo", "exhaustivo"],
            "num_sections": 10,
            "description": "Cobertura exhaustiva y especialización avanzada",
            "depth": "muy profunda",
            "focus": "dominio completo y técnicas de vanguardia"
        }
    }

    # Nivel por defecto
    level = "principiante"

    # Convertir el prompt a minúsculas para la comparación
    prompt_lower = prompt.lower()

    # Detectar el nivel en el prompt
    for level_name, level_info in levels.items():
        if any(keyword in prompt_lower for keyword in level_info["keywords"]):
            level = level_name
            break

    # Extraer el tema principal del prompt
    topic_extraction_prompt = f"""
    Analiza el siguiente prompt y extrae el tema principal del curso. 
    El tema principal debe ser una palabra o frase corta que identifique específicamente de qué trata el curso.
    Por ejemplo:
    - "curso de Java para principiantes porque quiero aprender programación" -> "Java"
    - "quiero aprender a tocar la guitarra desde cero" -> "Guitarra"
    - "curso de cocina italiana para principiantes" -> "Cocina italiana"

    Prompt: {prompt}

    Responde SOLO con el tema principal, sin explicaciones adicionales.
    """

    try:
        topic = call_google_generative_api_for_text(topic_extraction_prompt).strip()
    except Exception as e:
        print(f"Error extrayendo tema principal: {e}")
        # Fallback: eliminar palabras clave de nivel del prompt
        topic = prompt
        for level_info in levels.values():
            for keyword in level_info["keywords"]:
                topic = topic.replace(keyword, "").strip()

    # Obtener la configuración del nivel seleccionado
    level_config = levels[level]
    max_workers_outline = min(32, (os.cpu_count() or 1) * 2)  # Para paralelizar llamadas a IA aquí

    try:
        # Preparar los prompts para introducción y contenido
        intro_system_message = f"""
        Eres un experto en educación. Tu tarea es crear una introducción general y motivadora para un curso de nivel {level} sobre {topic}.

        La introducción debe:
        1. Ser breve y atractiva
        2. Explicar por qué es importante aprender {topic}
        3. Describir el enfoque del curso
        4. Motivar a los estudiantes
        5. No entrar en detalles técnicos (esos irán en las secciones)

        IMPORTANTE: No incluyas ninguna referencia a videos, URLs o contenido multimedia en la respuesta.

        Proporciona la respuesta en formato JSON:
        {{
            "introduction": "Texto de la introducción"
        }}
        """
        intro_user_message = f"Crea una introducción para un curso sobre: {topic}"
        intro_full_prompt = f"{intro_system_message}\n\nUSER QUESTION:\n{intro_user_message}"  # Combinar

        content_system_message = f"""
        Eres un experto en diseño de cursos educativos. Tu tarea es crear un esquema detallado para un curso de nivel {level} sobre {topic}.

        El curso debe incluir:
        1. Un título atractivo y descriptivo que refleje el nivel {level}
        2. {level_config['num_sections']} secciones principales, cada una con:
           - Título descriptivo que incluya el tema principal ({topic})
           - Descripción detallada del contenido
        3. Objetivos de aprendizaje específicos al nivel {level}
        4. Requisitos previos apropiados para el nivel

        Para un curso de nivel {level}, asegúrate de:
        - {level_config['description']}
        - Mantener una profundidad {level_config['depth']} en los temas
        - Enfocarse en {level_config['focus']}
        - {f"Incluir ejercicios prácticos y proyectos" if level != "principiante" else "Incluir ejemplos simples y ejercicios guiados"}
        - {f"Cubrir temas especializados y técnicas avanzadas" if level in ["avanzado", "maestro"] else "Mantener un enfoque en conceptos fundamentales"}

        IMPORTANTE: 
        1. No incluyas ninguna referencia a videos, URLs o contenido multimedia en la respuesta.
        2. Cada título de sección DEBE incluir el tema principal ({topic}) para asegurar que los videos encontrados sean relevantes.

        Proporciona la respuesta en formato JSON con la siguiente estructura:
        {{
            "title": "Título del curso",
            "sections": [
                {{
                    "title": "Título de la sección (incluyendo {topic})",
                    "description": "Descripción detallada"
                }}
            ],
            "learningOutcomes": ["Objetivo 1", "Objetivo 2", ...],
            "requirements": ["Requisito 1", "Requisito 2", ...],
            "level": "{level}",
            "level_description": "{level_config['description']}",
            "total_sections": {level_config['num_sections']}
        }}
        """
        content_user_message = f"Crea un curso sobre: {topic}"
        content_full_prompt = f"{content_system_message}\n\nUSER QUESTION:\n{content_user_message}"  # Combinar

        intro_data = None
        content_data = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_outline) as executor:
            future_intro = executor.submit(call_google_generative_api_for_text, intro_full_prompt)
            future_content = executor.submit(call_google_generative_api_for_text, content_full_prompt)

            try:
                intro_response_text = future_intro.result()
                intro_response_text = re.sub(r'^```json\s*', '', intro_response_text).strip()
                intro_response_text = re.sub(r'\s*```$', '', intro_response_text).strip()
                intro_data = json.loads(intro_response_text)
            except Exception as e_intro:
                print(f"Error generando introducción del curso en paralelo: {e_intro}")
                # Podríamos decidir si continuar sin introducción o lanzar el error
                raise  # Por ahora, relanzamos si falla la introducción

            try:
                content_response_text = future_content.result()
                content_response_text = re.sub(r'^```json\s*', '', content_response_text).strip()
                content_response_text = re.sub(r'\s*```$', '', content_response_text).strip()
                content_data = json.loads(content_response_text)
            except Exception as e_content:
                print(f"Error generando contenido del curso en paralelo: {e_content}")
                # Podríamos decidir si continuar sin contenido o lanzar el error
                raise  # Por ahora, relanzamos si falla el contenido

        # Combinar la introducción con el contenido del curso
        course_outline = {
            **content_data,  # Asegurarse que content_data no sea None
            "introduction": intro_data["introduction"],  # Asegurarse que intro_data no sea None
            "extracted_topic": topic
        }

        # Generar queries de búsqueda mejoradas
        course_outline["searchQueries"] = {
            "introductory": f"introducción a {topic} para {level}s",
            **{f"section{i}": f"{section['title']} {topic} tutorial {level}" for i, section in
               enumerate(course_outline["sections"])}
        }

        return course_outline

    except Exception as e:
        print(f"Error generando esquema del curso: {e}")
        raise


# --- Resto de las funciones SIN CAMBIOS ---

def calculate_video_score(video_details, snippet, statistics, days_since_published, total_minutes, section_content=""):
    """Calcula la puntuación de un video basada en múltiples criterios"""
    try:
        # 1. Relevancia (35%) - Ahora usando similitud de texto
        relevance_score = 0.0
        if section_content:
            # Combinar título y descripción del video
            video_text = f"{snippet.get('title', '')} {snippet.get('description', '')}"

            # Crear vectorizador TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english')
            try:
                # Convertir textos a vectores TF-IDF
                tfidf_matrix = vectorizer.fit_transform([video_text, section_content])
                # Calcular similitud coseno
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                relevance_score = float(similarity)
            except Exception as e:
                print(f"Error calculando similitud: {e}")
                relevance_score = 0.5  # Valor por defecto si hay error
        else:
            relevance_score = 0.5  # Valor por defecto si no hay contenido de sección

        # 2. Calidad del video (20%)
        quality_score = 0
        definition = video_details.get('contentDetails', {}).get('definition')
        if definition == 'hd':
            quality_score = 0.8
        elif definition == 'sd':
            quality_score = 0.4
        else:
            quality_score = 0.3

        # 3. Engagement (20%)
        views = int(statistics.get("viewCount", 0))
        likes = int(statistics.get("likeCount", 0))
        comments = int(statistics.get("commentCount", 0))

        engagement_score = 0
        if views > 100:
            like_ratio = (likes / views)
            comment_ratio = (comments / views)
            engagement_score = min(1.0, (like_ratio * 5) * 0.6 + (comment_ratio * 20) * 0.4)

        # 4. Actualidad (15%)
        recency_score = 1.0 if days_since_published <= 365 else (
            0.5 if days_since_published <= 730 else 0.2)

        # 5. Duración (10%) - Ideal entre 5 y 25 mins
        duration_score = 1.0 if 5 <= total_minutes <= 25 else (
            0.5 if total_minutes < 5 else 0.7)

        # Verificar qué scores están disponibles y redistribuir pesos
        available_scores = {}
        total_weight = 0.0

        if relevance_score > 0:
            available_scores['relevance'] = (relevance_score, 0.35)
            total_weight += 0.35

        if quality_score > 0:
            available_scores['quality'] = (quality_score, 0.20)
            total_weight += 0.20

        if engagement_score > 0:
            available_scores['engagement'] = (engagement_score, 0.20)
            total_weight += 0.20

        if recency_score > 0:
            available_scores['recency'] = (recency_score, 0.15)
            total_weight += 0.15

        if duration_score > 0:
            available_scores['duration'] = (duration_score, 0.10)
            total_weight += 0.10

        # Redistribuir pesos proporcionalmente si faltan algunos scores
        final_score = 0.0
        if available_scores:
            for score_name, (score_value, original_weight) in available_scores.items():
                adjusted_weight = original_weight / total_weight if total_weight > 0 else 0
                final_score += score_value * adjusted_weight

        # Imprimir puntajes en consola
        print(f"=== PUNTAJES DEL VIDEO ===")
        print(f"Relevance Score: {relevance_score:.3f}")
        print(f"Quality Score: {quality_score:.3f}")
        print(f"Engagement Score: {engagement_score:.3f}")
        print(f"Recency Score: {recency_score:.3f}")
        print(f"Duration Score: {duration_score:.3f}")
        print(f"FINAL SCORE: {final_score:.3f}")
        print(f"========================")

        return final_score

    except Exception as e:
        print(f"Error calculando score del video: {e}")
        return 0.0  # Retornar 0 si hay algún error en el cálculo


def get_video_comments(video_id, max_results=50):
    """Obtiene los comentarios de un video (SIN CAMBIOS)"""
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText",
            order="relevance"  # Mantener orden original
        )
        response = request.execute()

        comments = []
        for item in response['items']:
            # Acceso original a los datos del comentario
            comment_snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                "author": comment_snippet['authorDisplayName'],
                "text": comment_snippet['textDisplay']  # Usar textDisplay como en original
                # "publishedAt": comment_snippet['publishedAt'], # No estaban en el original
                # "likeCount": comment_snippet['likeCount'] # No estaban en el original
            })
        return comments
    except HttpError as e:
        # Mantener manejo de error original
        if e.resp.status == 403 and 'commentsDisabled' in str(e.content):  # Checar e.content es más robusto
            print(f"Los comentarios están deshabilitados para el video {video_id}")
            return []
        else:
            print(f"Error HTTP {e.resp.status} al obtener comentarios para {video_id}: {e}")
            return []
    except Exception as e:
        print(f"Error inesperado obteniendo comentarios para {video_id}: {e}")
        return []


def verificar_transcripcion_disponible(video_id):
    """Verifica si un video tiene transcripción disponible sin descargarla"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Verificar si existe transcripción en español
        if transcript_list.find_transcript(['es']):
            return True
        # Si no hay en español, verificar si hay en otros idiomas
        available_transcripts = transcript_list.find_manually_created_transcript()
        return bool(available_transcripts)
    except Exception as e:
        print(f"Error verificando transcripción para video {video_id}: {e}")
        return False


def search_youtube_videos(query, max_results=4, section_content="", used_video_ids=None):
    """Search for YouTube videos based on query and return the best match"""
    if used_video_ids is None:
        used_video_ids = set()

    # Determinar el número de hilos
    max_workers = min(32, (os.cpu_count() or 1) * 2)

    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        # Primera búsqueda con licencia YouTube, retrieving up to 10 videos (antes 50)
        request_youtube = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            videoLicense="creativeCommon",
            maxResults=10,  # Cambiado de 50 a 10
            relevanceLanguage="es",
            videoDuration="medium",
            order="relevance"
        )
        response = request_youtube.execute()

        videos = []
        all_videos_data = []
        if response.get("items"):
            # Obtener IDs de video para metadatos adicionales
            video_ids = [item["id"]["videoId"] for item in response["items"]]

            # Obtener estadísticas y detalles del contenido
            video_details = youtube.videos().list(
                part="statistics,contentDetails,snippet",
                id=",".join(video_ids)
            ).execute()

            # Crear mapa de ID a detalles
            details_map = {item["id"]: item for item in video_details["items"]}

            # Procesar cada video
            for item in response["items"]:
                video_id = item["id"]["videoId"]

                # Verificar si tiene transcripción disponible
                #if not verificar_transcripcion_disponible(video_id):
                #   print(f"Video {video_id} descartado: No tiene transcripción disponible")
                #   continue

                video_details = details_map.get(video_id, {})
                statistics = video_details.get("statistics", {})
                content_details = video_details.get("contentDetails", {})
                snippet = video_details.get("snippet", {})

                # Extraer duración
                duration = content_details.get("duration", "PT0M0S")
                minutes_match = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
                if minutes_match:
                    hours = int(minutes_match.group(1) or 0)
                    minutes = int(minutes_match.group(2) or 0)
                    seconds = int(minutes_match.group(3) or 0)
                    total_minutes = (hours * 60) + minutes + (seconds / 60)
                    duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes} min"
                else:
                    total_minutes = 0
                    duration_str = "Desconocido"

                # Calcular días desde publicación
                published_at = datetime.strptime(snippet.get("publishedAt", ""), "%Y-%m-%dT%H:%M:%SZ")
                days_since_published = (datetime.now() - published_at).days

                # Calcular puntuación inicial
                score = calculate_video_score(video_details, snippet, statistics, days_since_published, total_minutes)

                # Collect all relevant video data
                video_data = {
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "videoUrl": f"https://www.youtube.com/embed/{video_id}",
                    "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
                    "channelTitle": item["snippet"]["channelTitle"],
                    "publishedAt": item["snippet"]["publishedAt"],
                    "views": int(statistics.get("viewCount", 0)),
                    "likes": int(statistics.get("likeCount", 0)),
                    "comments": int(statistics.get("commentCount", 0)),
                    "duration": duration_str,
                    "score": score,
                    "videoId": video_id,
                    "totalMinutes": total_minutes,
                    "full_snippet": snippet,
                    "full_statistics": statistics,
                    "full_contentDetails": content_details
                }

                all_videos_data.append(video_data)

            # Ordenar los videos por score y tomar los 5 mejores (antes 10)
            sorted_videos = sorted(all_videos_data, key=lambda x: x["score"], reverse=True)
            top_5_videos = sorted_videos[:5]  # Cambiado de top_10_videos a top_5_videos y slice a :5

            # Procesar las transcripciones de los 5 mejores videos en paralelo
            videos_with_data = []  # ✅ CAMBIO: Cambiar nombre para reflejar que no todos tienen transcripción
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_video = {executor.submit(get_video_transcript, video["videoId"]): video for video in
                                   top_5_videos}
                for future in concurrent.futures.as_completed(future_to_video):
                    video = future_to_video[future]
                    try:
                        transcript = future.result()
                        # ✅ CAMBIO: Incluir el video independientemente de si tiene transcripción
                        video["processed_transcript"] = transcript if transcript else ""
                        videos_with_data.append(video)

                        if transcript:
                            print(f"Video {video['videoId']}: Transcripción obtenida exitosamente.")
                        else:
                            print(f"Video {video['videoId']}: Sin transcripción, pero incluido para análisis.")
                    except Exception as exc:
                        print(f"Error obteniendo transcripción para {video['videoId']} en paralelo: {exc}")
                        # ✅ CAMBIO: Incluir el video incluso si hay error
                        video["processed_transcript"] = ""
                        videos_with_data.append(video)

            return videos_with_data  # ✅ CAMBIO: Devolver todos los videos, con o sin transcripción

        return []

    except Exception as e:
        print(f"Error searching YouTube videos: {e}")
        return []


def limpiar_comentarios(comentarios):
    """Limpia y normaliza una lista de comentarios."""
    comentarios_limpios = set()
    for comentario in comentarios:
        texto = comentario.get("text", "")
        # Eliminar emojis
        texto = emoji.replace_emoji(texto, replace="")
        # Eliminar símbolos y caracteres especiales
        texto = re.sub(r'[^\w\s]', '', texto)
        # Convertir a minúsculas y eliminar espacios extra
        texto = texto.lower().strip()
        if texto:
            comentarios_limpios.add(texto)
    return list(comentarios_limpios)


def analizar_sentimiento_comentarios(comentarios_limpios, analyzer):
    """Analiza el sentimiento de los comentarios usando pysentimiento y retorna resumen."""
    if not comentarios_limpios:
        return {
            "resumen": "Sin comentarios",
            "proporcion": {"POS": 0, "NEU": 0, "NEG": 0},
            "total": 0
        }
    resultados = {"POS": 0, "NEU": 0, "NEG": 0}
    for comentario in comentarios_limpios:
        resultado = analyzer.predict(comentario)
        resultados[resultado.output.upper()] += 1
    total = len(comentarios_limpios)
    proporcion = {k: v / total for k, v in resultados.items()}
    # Resumen general
    mayor = max(proporcion, key=proporcion.get)
    resumen = {
        "POS": "mayoría positiva",
        "NEU": "mayoría neutral",
        "NEG": "mayoría negativa"
    }[mayor]
    return {
        "resumen": resumen,
        "proporcion": proporcion,
        "total": total
    }


def analyze_video_content(video_id, section_content, used_video_ids, course_topic, video_youtube_title,
                          processed_transcript_text=None, analyzer=analyzer_sentimiento):
    """Analiza si el contenido del video es relevante para la sección usando Google API y sentimiento de comentarios."""
    if video_id in used_video_ids:
        return False, "Video ya usado en otra sección", None

    analysis_response_json = None
    try:
        transcript = processed_transcript_text
        if not transcript:
            print(f"No se proveyó transcripción para {video_id}, obteniéndola ahora...")
            transcript = get_video_transcript(video_id)
            # ✅ CAMBIO: No verificar si transcript es None, usar string vacío si es necesario
            if transcript is None:
                transcript = ""

        # ✅ CAMBIO: Manejar transcripción vacía en el prompt
        transcript_text = transcript if transcript.strip() else "No hay transcripción disponible para este video."

        # Obtener y procesar comentarios
        comentarios = get_video_comments(video_id, max_results=50)
        comentarios_limpios = limpiar_comentarios(comentarios)
        sentimiento = analizar_sentimiento_comentarios(comentarios_limpios, analyzer)

        # ✅ CAMBIO: Modificar el system_message para manejar videos sin transcripción
        system_message = f"""
Eres un experto en análisis de contenido educativo. Tu tarea es determinar si el contenido de un video (considerando su título original de YouTube, su transcripción si está disponible, y el sentimiento de los comentarios) es relevante para una sección específica de un curso sobre '{course_topic}'.

Debes analizar:
1. Si el contenido del video (título y transcripción si disponible) coincide con el tema de la sección descrito abajo.
2. Si el nivel de profundidad es apropiado.
3. Si la información es precisa y relevante para '{course_topic}'.
4. Si el video (título y transcripción si disponible) cubre los conceptos principales mencionados en la descripción de la sección, en el contexto de '{course_topic}'.
5. Crucialmente, verifica que el video (título y transcripción si disponible) trate específicamente sobre '{course_topic}' y no sobre temas relacionados pero distintos.
6. Analiza el sentimiento general de los comentarios: si la mayoría es negativa, penaliza el video; si es positiva, súmalo al score.

IMPORTANTE: 
- Si NO hay transcripción disponible, basa tu análisis PRINCIPALMENTE en el título del video y los comentarios.
- El título debe ser del tema correcto (por ejemplo, si la sección es de variables en Java, rechaza si el título es de variables en C++ o C).
- Si no hay transcripción, sé más flexible con el score pero mantén la relevancia del título.

Asigna un score de relevancia total considerando:
- Si HAY transcripción: Título: 40%, Transcripción: 30%, Sentimiento de comentarios: 30%
- Si NO HAY transcripción: Título: 70%, Sentimiento de comentarios: 30%

Responde con un JSON que contenga:
{{
    'is_relevant': true/false,
    'confidence_score': número entre 0 y 1,  # Score total considerando los pesos
    'reason': "explicación detallada de por qué el video es o no relevante, considerando específicamente el tema del curso '{course_topic}', el título del video, su transcripción (si disponible) y el sentimiento de los comentarios.",
    'topic_match': true/false, # True si el video (título y transcripción si disponible) es sobre '{course_topic}', False en caso contrario.
    'topic_match_score': número entre 0 y 1, # Confianza en que el video (título y transcripción si disponible) trata sobre '{course_topic}'
    'has_transcript': true/false, # Indica si el video tiene transcripción disponible
    'sentiment_summary': "{sentimiento['resumen']}",
    'sentiment_proportion': {sentimiento['proporcion']},
    'sentiment_total_comments': {sentimiento['total']}
}}
"""

        user_message = f"""
Analiza si el siguiente contenido de video es relevante para esta sección del curso:

Título y descripción de la sección:
{section_content}

Título original del video de YouTube:
{video_youtube_title}

Transcripción del video:
{transcript_text}
"""
        analysis_full_prompt = f"{system_message}\n\nUSER QUESTION:\n{user_message}"

        # Llamar a la API de Google
        analysis_response_text = call_google_generative_api_for_text(analysis_full_prompt)
        analysis_response_text = re.sub(r'^```json\s*', '', analysis_response_text).strip()
        analysis_response_text = re.sub(r'\s*```$', '', analysis_response_text).strip()
        analysis_response_json = json.loads(analysis_response_text)

        # ✅ CAMBIO: Ajustar los umbrales si no hay transcripción
        has_transcript = analysis_response_json.get("has_transcript", True)
        confidence_threshold = 0.6 if not has_transcript else 0.7  # Umbral más bajo sin transcripción

        # Verificar si el tema coincide específicamente
        if not analysis_response_json.get("topic_match", False):
            print(f"Análisis video {video_id}: Rechazado por tema incorrecto. Razón: {analysis_response_json.get('reason', 'N/A')}")
            return False, f"El video no trata específicamente del tema correcto: {analysis_response_json.get('reason', 'Sin razón específica')}", analysis_response_json

        # Si el tema coincide pero la confianza es baja, aún podríamos aceptarlo como último recurso
        if analysis_response_json.get("confidence_score", 0) < confidence_threshold:
            if analysis_response_json.get("topic_match_score", 0) >= 0.8:  # Si el tema coincide bien
                print(f"Análisis video {video_id}: Aceptado con baja confianza pero tema correcto (Confianza: {analysis_response_json.get('confidence_score', 0):.2f})")
                return True, f"Video aceptado como alternativa. {analysis_response_json.get('reason', 'Análisis exitoso')}", analysis_response_json
            else:
                print(f"Análisis video {video_id}: Rechazado por baja confianza ({analysis_response_json.get('confidence_score', 0):.2f}). Razón: {analysis_response_json.get('reason', 'N/A')}")
                return False, f"Contenido no suficientemente relevante (Confianza < {confidence_threshold}): {analysis_response_json.get('reason', 'Sin razón específica')}", analysis_response_json

        if analysis_response_json.get("is_relevant", False):
            transcript_status = "con transcripción" if has_transcript else "sin transcripción"
            print(f"Análisis video {video_id}: Aprobado {transcript_status} (Confianza: {analysis_response_json.get('confidence_score', 0):.2f}). Razón: {analysis_response_json.get('reason', 'N/A')}")
            return True, f"Video aprobado por análisis de contenido. {analysis_response_json.get('reason', 'Análisis exitoso')}", analysis_response_json
        else:
            print(f"Análisis video {video_id}: Rechazado por IA. Razón: {analysis_response_json.get('reason', 'N/A')}")
            return False, f"Contenido no relevante según análisis: {analysis_response_json.get('reason', 'IA determinó no relevante')}", analysis_response_json

    except Exception as e:
        print(f"Error en el análisis de contenido para {video_id}: {e}")
        if 'analysis_response_text' in locals() and isinstance(e, json.JSONDecodeError):
            print("Texto recibido (análisis):", analysis_response_text)
        return False, f"Error durante el análisis de contenido: {str(e)}", analysis_response_json


def process_section_parallel(section_data, section_index, topic, used_video_ids, analyzer):
    """Procesa una sección del curso en paralelo, incluyendo búsqueda y análisis de videos."""
    section_id = section_index + 1
    print(f"Procesando Sección {section_id}: {section_data.get('title', 'Sin Título')}")

    section_title_full = section_data.get('title', '')

    # Lógica de simplificación de la consulta principal
    simplified_title_parts = section_title_full.split(':')[0].strip()
    if not simplified_title_parts or len(
            simplified_title_parts.split()) < 2 or simplified_title_parts == section_title_full:
        simplified_title_parts = ' '.join(section_title_full.split()[:4]).strip()

    # Construir la consulta principal simplificada con el topic
    if topic.lower() in simplified_title_parts.lower():
        section_query = simplified_title_parts
    else:
        section_query = f"{simplified_title_parts} {topic}"
    section_query = section_query.strip()

    print(f"Buscando videos para Sección {section_id} con query principal simplificada: '{section_query}'")
    all_section_videos_data = search_youtube_videos(
        section_query,
        section_content=section_data.get("description", ""),
        used_video_ids=used_video_ids
    )

    if not all_section_videos_data:
        # Alternativa 1: Usar el título completo de la sección + topic
        alt_query_1 = f"{section_title_full} {topic}".strip()
        print(
            f"Query simplificada falló. Intentando alternativa 1 (título completo) para Sección {section_id}: '{alt_query_1}'")
        all_section_videos_data = search_youtube_videos(
            alt_query_1,
            section_content=section_data.get("description", ""),
            used_video_ids=used_video_ids
        )

        if not all_section_videos_data:
            # Alternativa 2: Usar solo el topic y el nivel
            alt_query_2 = f"{topic} {section_data.get('level', 'principiante')}".strip()
            print(
                f"Alternativa 1 falló. Intentando alternativa 2 (solo tema y nivel) para Sección {section_id}: '{alt_query_2}'")
            all_section_videos_data = search_youtube_videos(
                alt_query_2,
                section_content=section_data.get("description", ""),
                used_video_ids=used_video_ids
            )

    section_result = {
        "id": section_id,
        "title": section_data.get("title", f"Sección {section_id}"),
        "content": section_data.get("description", "Contenido no disponible."),
        "videoUrl": None,
        "duration": "N/A",
        "classes": 1,
        "videoId": None,
        "videoTitle": None,
        "videos_data": all_section_videos_data
    }

    return section_result


def get_comments_parallel(video_id, max_results=50):
    """Obtiene comentarios de un video en paralelo."""
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText",
            order="relevance"
        )
        response = request.execute()

        comments = []
        for item in response['items']:
            comment_snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                "author": comment_snippet['authorDisplayName'],
                "text": comment_snippet['textDisplay']
            })
        return comments
    except HttpError as e:
        if e.resp.status == 403 and 'commentsDisabled' in str(e.content):
            print(f"Los comentarios están deshabilitados para el video {video_id}")
            return []
        else:
            print(f"Error HTTP {e.resp.status} al obtener comentarios para {video_id}: {e}")
            return []
    except Exception as e:
        print(f"Error inesperado obteniendo comentarios para {video_id}: {e}")
        return []


@app.route("/solicitar_cursos", methods=["POST"])
def solicitar_cursos():
    try:
        data = request.json
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "El prompt no puede estar vacío"}), 400

        max_workers = min(32, (os.cpu_count() or 1) * 2)

        if not GOOGLE_API_KEY or not YOUTUBE_API_KEY:
            return jsonify({"error": "Error de configuración: Faltan claves API en el servidor."}), 500

        print(f"\n=== Iniciando generación de curso para prompt: '{prompt}' ===")

        try:
            course_outline = get_course_outline(prompt)
            print("Esquema del curso generado con éxito.")
        except Exception as e:
            print(f"Fallo crítico: No se pudo generar el esquema del curso. Error: {e}")
            return jsonify({"error": f"Error al generar la estructura base del curso: {str(e)}"}), 500

        topic = course_outline.get("extracted_topic", prompt.split(' ')[0])
        print(f"Tema principal extraído para la búsqueda de videos: '{topic}'")

        course_id = f"course_{int(time.time())}"
        used_video_ids = set()

        # Procesar secciones en paralelo
        print("\n--- Procesando secciones en paralelo ---")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_section = {
                executor.submit(
                    process_section_parallel,
                    section,
                    i,
                    topic,
                    used_video_ids,
                    analyzer_sentimiento
                ): (i, section)
                for i, section in enumerate(course_outline.get("sections", []))
            }

            sections = []
            for future in concurrent.futures.as_completed(future_to_section):
                try:
                    section_result = future.result()
                    sections.append(section_result)
                except Exception as exc:
                    print(f"Error procesando sección en paralelo: {exc}")

        # Ordenar secciones por ID
        sections.sort(key=lambda x: x["id"])

        # Procesar videos de cada sección en paralelo
        print("\n--- Analizando videos de secciones en paralelo ---")
        for section in sections:
            if not section.get("videos_data"):
                continue

            approved_section_videos = []
            all_analyzed_videos_with_ai_results = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_video_section = {
                    executor.submit(
                        analyze_video_content,
                        video_data["videoId"],
                        section.get("content", ""),
                        used_video_ids,
                        topic,
                        video_data.get("title"),
                        video_data.get("processed_transcript"),
                        analyzer=analyzer_sentimiento
                    ): video_data
                    for video_data in section["videos_data"]
                }

                for future in concurrent.futures.as_completed(future_to_video_section):
                    video_data = future_to_video_section[future]
                    try:
                        is_approved, reason, analysis_details = future.result()
                        if is_approved:
                            approved_section_videos.append(video_data)
                        if analysis_details:
                            all_analyzed_videos_with_ai_results.append((video_data, analysis_details))
                    except Exception as exc:
                        print(f"Error analizando video de sección {video_data['videoId']} en paralelo: {exc}")

            # Seleccionar el mejor video para la sección
            if approved_section_videos:
                best_video = sorted(approved_section_videos, key=lambda x: x["score"], reverse=True)[0]
                used_video_ids.add(best_video["videoId"])
                section.update({
                    "duration": best_video["duration"],
                    "videoId": best_video["videoId"],
                    "videoTitle": best_video["title"],
                    "videoUrl": f"https://www.youtube.com/embed/{best_video['videoId']}"
                })
            elif all_analyzed_videos_with_ai_results:
                available_for_fallback = [
                    (vid_data, analysis)
                    for vid_data, analysis in all_analyzed_videos_with_ai_results
                    if vid_data["videoId"] not in used_video_ids
                ]

                if available_for_fallback:
                    available_for_fallback.sort(key=lambda x: x[1].get('confidence_score', 0), reverse=True)
                    best_fallback_video_data, best_fallback_analysis = available_for_fallback[0]

                    used_video_ids.add(best_fallback_video_data["videoId"])
                    section.update({
                        "duration": best_fallback_video_data["duration"],
                        "videoId": best_fallback_video_data["videoId"],
                        "videoTitle": best_fallback_video_data["title"],
                        "videoUrl": f"https://www.youtube.com/embed/{best_fallback_video_data['videoId']}"
                    })

            # Eliminar datos temporales
            section.pop("videos_data", None)

        # Añadir sección de evaluación final
        sections.append({
            "id": len(sections) + 1,
            "title": "Evaluación Final",
            "content": "Evalúa lo aprendido en el curso.",
            "videoUrl": None,
            "duration": "N/A",
            "classes": 1,
            "videoId": None,
            "videoTitle": None
        })

        # Calcular duración total y clases
        total_classes = len(sections)
        total_minutes_calculation = sum(
            float(section.get("duration", "0").split()[0])
            for section in sections
            if section.get("duration") != "N/A"
        )

        total_duration_str = "N/A"
        if total_minutes_calculation > 0:
            total_hours = int(total_minutes_calculation // 60)
            total_mins = int(total_minutes_calculation % 60)
            if total_hours > 0:
                total_duration_str = f"{total_hours}h {total_mins}m"
            else:
                total_duration_str = f"{total_mins}m"

        current_date = datetime.now().strftime("%m/%Y")
        course_data = {
            "title": course_outline.get("title", f"Curso sobre {prompt}"),
            "introduction": course_outline.get("introduction", "Introducción no disponible"),
            "instructor": "IA Professor",
            "rating": round(random.uniform(4.5, 4.9), 1),
            "students": random.randint(5000, 15000),
            "lastUpdated": current_date,
            "language": "Español",
            "totalDuration": total_duration_str,
            "totalLessons": total_classes,
            "sections": sections,
            "learningOutcomes": course_outline.get("learningOutcomes", []),
            "requirements": course_outline.get("requirements", []),
            "level": course_outline.get("level", "principiante"),
            "level_description": course_outline.get("level_description", "")
        }

        end_time = time.time()
        print(f"=== Generación de curso completada en {end_time :.2f} segundos ===")
        return jsonify(course_data)

    except Exception as e:
        print(f"Error generating course: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Ocurrió un error inesperado: {str(e)}"}), 500


# Add a health check endpoint (SIN CAMBIOS)
@app.route("/health", methods=["GET"])
def health_check():
    # Chequeo básico original
    youtube_ok = bool(YOUTUBE_API_KEY)
    # Ahora checa Google API Key
    google_ok = bool(GOOGLE_API_KEY)
    status = {
        "status": "ok" if youtube_ok and google_ok else "error",
        "dependencies": {
            "youtube_api": "configured" if youtube_ok else "missing_key",
            "google_generative_api": "configured" if google_ok else "missing_key"
        }
    }
    status_code = 200 if status["status"] == "ok" else 503
    return jsonify(status), status_code


# --- Ejecución Principal SIN CAMBIOS ---
if __name__ == "__main__":
    # Mantener forma original de correr la app
    port = int(os.environ.get("PORT", 5000))  # Puerto 5000 como en muchos ejemplos Flask
    app.run(host="0.0.0.0", port=port, debug=True)  # debug=True como en original