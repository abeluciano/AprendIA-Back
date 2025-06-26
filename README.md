# Sistema Generador de Cursos con IA

Un sistema automatizado que genera cursos educativos completos utilizando inteligencia artificial y contenido de YouTube. El sistema analiza prompts de usuario, crea estructuras de curso personalizadas y busca videos relevantes de YouTube con transcripciones para cada sección.

## 🚀 Características Principales

- **Generación Automática de Cursos**: Crea estructuras de curso completas basadas en prompts de usuario
- **Múltiples Niveles**: Soporte para cursos de nivel principiante, intermedio, avanzado y maestro
- **Búsqueda Inteligente de Videos**: Encuentra automáticamente videos de YouTube relevantes para cada sección
- **Análisis de Contenido con IA**: Utiliza Google Gemini para analizar la relevancia del contenido
- **Procesamiento de Transcripciones**: Extrae y procesa transcripciones de YouTube en español
- **Análisis de Sentimientos**: Evalúa comentarios de videos para determinar su calidad
- **Procesamiento Paralelo**: Optimizado para rendimiento con concurrencia

## 🛠️ Tecnologías Utilizadas

### Backend
- **Flask**: Framework web principal
- **Flask-CORS**: Manejo de CORS para frontend
- **Google Gemini API**: Generación de contenido con IA
- **YouTube Data API v3**: Búsqueda y metadatos de videos
- **YouTube Transcript API**: Extracción de transcripciones

### Procesamiento de Texto y ML
- **scikit-learn**: Análisis TF-IDF y similitud coseno
- **NLTK**: Procesamiento de lenguaje natural
- **spaCy**: Análisis lingüístico avanzado (español)
- **TextBlob**: Análisis de texto y detección de duplicados
- **pysentimiento**: Análisis de sentimientos en español
- **sumy**: Resumenes automáticos con TextRank

### Utilidades
- **python-dotenv**: Gestión de variables de entorno
- **requests**: Llamadas HTTP a APIs
- **emoji**: Procesamiento de emojis
- **spellchecker**: Corrección ortográfica

## 📋 Requisitos Previos

1. **Python 3.8+**
2. **Claves API requeridas**:
   - Google API Key (para Gemini)
   - YouTube Data API Key
3. **Modelo de spaCy español**:
   ```bash
   python -m spacy download es_core_news_sm
   ```

## ⚙️ Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/abeluciano/AprendIA-Back.git
   cd AprendIA-Back
   ```

2. **Crear entorno virtual**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install flask flask-cors python-dotenv requests
   pip install scikit-learn numpy nltk textblob spellchecker
   pip install youtube-transcript-api google-api-python-client
   pip install sumy spacy pysentimiento emoji
   python -m spacy download es_core_news_sm
   ```

4. **Configurar variables de entorno**:
   Crear archivo `.env`:
   ```env
   GOOGLE_API_KEY=tu_google_api_key_aqui
   YOUTUBE_API_KEY=tu_youtube_api_key_aqui
   GEMINI_MODEL_NAME=tu_version_de_gemini_a_usar
   ```

5. **Ejecutar la aplicación**:
   ```bash
   python app.py
   ```

## 🔧 Configuración de APIs

### Google Gemini API
1. Ir a [Google AI Studio](https://makersuite.google.com/)
2. Crear una nueva API key
3. Agregar la key a tu archivo `.env`

### YouTube Data API
1. Ir a [Google Cloud Console](https://console.cloud.google.com/)
2. Crear un proyecto y habilitar YouTube Data API v3
3. Crear credenciales (API Key)
4. Agregar la key a tu archivo `.env`

## 📡 Endpoints de la API

### `POST /solicitar_cursos`
Genera un curso completo basado en un prompt de usuario.

**Request Body**:
```json
{
  "prompt": "curso de Python para principiantes"
}
```

**Response**:
```json
{
  "title": "Python para Principiantes: Fundamentos de Programación",
  "introduction": "Introducción motivadora al curso...",
  "instructor": "IA Professor",
  "rating": 4.7,
  "students": 8500,
  "language": "Español",
  "totalDuration": "3h 45m",
  "totalLessons": 8,
  "level": "principiante",
  "sections": [
    {
      "id": 1,
      "title": "Introducción a Python",
      "content": "Descripción detallada...",
      "videoUrl": "https://www.youtube.com/embed/VIDEO_ID",
      "duration": "25 min",
      "videoTitle": "Título del video"
    }
  ],
  "learningOutcomes": [...],
  "requirements": [...]
}
```

### `GET /health`
Verifica el estado del servicio y las APIs configuradas.

**Response**:
```json
{
  "status": "ok",
  "dependencies": {
    "youtube_api": "configured",
    "google_generative_api": "configured"
  }
}
```

## 🏗️ Arquitectura del Sistema

### Flujo Principal

1. **Recepción del Prompt**: El usuario envía un prompt describiendo el curso deseado
2. **Análisis de Nivel**: Detecta automáticamente el nivel del curso (principiante, intermedio, avanzado, maestro)
3. **Generación de Estructura**: Utiliza Gemini para crear el esquema del curso
4. **Búsqueda de Videos**: Busca videos relevantes en YouTube para cada sección
5. **Análisis de Contenido**: Evalúa la relevancia de cada video usando IA
6. **Procesamiento de Transcripciones**: Extrae y procesa transcripciones cuando están disponibles
7. **Análisis de Sentimientos**: Evalúa comentarios para determinar la calidad del video
8. **Selección Final**: Elige el mejor video para cada sección

### Componentes Principales

#### Generador de Esquemas (`get_course_outline`)
- Detecta el nivel del curso
- Genera estructura personalizada
- Crea objetivos de aprendizaje
- Define requisitos previos

#### Motor de Búsqueda (`search_youtube_videos`)
- Busca videos con licencia Creative Commons
- Aplica filtros de duración y relevancia
- Calcula scores basados en múltiples métricas
- Procesa transcripciones en paralelo

#### Analizador de Contenido (`analyze_video_content`)
- Evalúa relevancia temática
- Analiza sentimientos de comentarios
- Considera disponibilidad de transcripciones
- Asigna scores de confianza

#### Procesador de Transcripciones (`procesar_transcripcion`)
- Limpia y normaliza texto
- Corrige errores comunes
- Elimina redundancias
- Extrae contenido más relevante usando TextRank

## 🎯 Algoritmo de Scoring

### Criterios de Evaluación de Videos

1. **Relevancia (35%)**: Similitud TF-IDF entre contenido del video y sección
2. **Calidad (20%)**: Definición del video (HD vs SD)
3. **Engagement (20%)**: Ratio de likes y comentarios
4. **Actualidad (15%)**: Qué tan reciente es el video
5. **Duración (10%)**: Duración óptima entre 5-25 minutos

### Análisis de Sentimientos
- **Positivo**: Aumenta el score del video
- **Neutral**: No afecta el score
- **Negativo**: Penaliza el video

## 🔄 Procesamiento Paralelo

El sistema utiliza `concurrent.futures.ThreadPoolExecutor` para:
- Búsqueda simultánea de videos para múltiples secciones
- Procesamiento paralelo de transcripciones
- Análisis concurrente de contenido con IA
- Obtención paralela de comentarios

## 🚨 Manejo de Errores

- **Transcripciones no disponibles**: El sistema continúa funcionando basándose en títulos y comentarios
- **Límites de API**: Implementa timeouts y manejo de errores HTTP
- **Videos sin comentarios**: Maneja casos donde los comentarios están deshabilitados
- **Fallbacks**: Sistema de respaldo para consultas de búsqueda


## 📊 Limitaciones Conocidas

- Dependiente de la disponibilidad de contenido en YouTube
- Limitado a videos con licencia Creative Commons
- Requiere conexión a internet estable
- Procesamiento intensivo para cursos con muchas secciones
