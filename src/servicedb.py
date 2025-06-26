from flask import request, jsonify, redirect, url_for, session, Blueprint
from flask_cors import cross_origin
from authlib.integrations.flask_client import OAuth
from datetime import datetime
import secrets
import json
from dotenv import load_dotenv
import os
from .db import DatabaseConnection  # Importa tu clase singleton

# Cargar variables de entorno
load_dotenv()



# Crear un blueprint en lugar de una aplicación
blueprint = Blueprint('auth', __name__)

# Configuración de OAuth con Google
oauth = OAuth()
google = oauth.register(
    name='google',
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    access_token_url='https://oauth2.googleapis.com/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v3/',
    userinfo_endpoint='https://openidconnect.googleapis.com/v1/userinfo',
    jwks_uri='https://www.googleapis.com/oauth2/v3/certs',
    client_kwargs={'scope': 'openid email profile'},
)

# Instancia del singleton de base de datos
db = DatabaseConnection()


def generate_token():
    """Genera un token único para el usuario"""
    return secrets.token_urlsafe(32)


def create_or_update_user(user_info):
    """Crea o actualiza un usuario basado en la información de Google"""
    try:
        # Verificar si el usuario ya existe
        existing_user = db.execute_query(
            "SELECT id_usuario, token FROM usuarios WHERE correo = %s",
            (user_info['email'],)
        )

        if existing_user:
            # Usuario existe, actualizar token y datos
            user_id = existing_user[0][0]
            new_token = generate_token()

            db.execute_query(
                """UPDATE usuarios 
                   SET token = %s, nombre_completo = %s, actualizado_en = CURRENT_TIMESTAMP 
                   WHERE id_usuario = %s""",
                (new_token, user_info['name'], user_id),
                fetch=False
            )

            return {'id_usuario': user_id, 'token': new_token, 'is_new': False}
        else:
            # Usuario nuevo, crear registro
            new_token = generate_token()
            username = user_info['email'].split('@')[0]  # Usar parte del email como username

            # Verificar que el username sea único
            counter = 1
            original_username = username
            while True:
                existing_username = db.execute_query(
                    "SELECT id_usuario FROM usuarios WHERE nombre_usuario = %s",
                    (username,)
                )
                if not existing_username:
                    break
                username = f"{original_username}_{counter}"
                counter += 1

            # Insertar nuevo usuario
            result = db.execute_query(
                """INSERT INTO usuarios (nombre_usuario, correo, contrasena_hash, nombre_completo, token) 
                   VALUES (%s, %s, %s, %s, %s) RETURNING id_usuario""",
                (username, user_info['email'], 'google_auth', user_info['name'], new_token),
                fetch=True
            )

            return {'id_usuario': result[0][0], 'token': new_token, 'is_new': True}

    except Exception as e:
        print(f"Error en create_or_update_user: {e}")
        raise


def verify_token(token):
    """Verifica si el token es válido y retorna el usuario"""
    try:
        result = db.execute_query(
            "SELECT id_usuario, nombre_usuario, correo, nombre_completo FROM usuarios WHERE token = %s AND esta_activo = true",
            (token,)
        )
        return result[0] if result else None
    except Exception as e:
        print(f"Error en verify_token: {e}")
        return None


def require_auth():
    """Decorator para requerir autenticación"""

    def decorator(f):
        def wrapper(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token or not token.startswith('Bearer '):
                return jsonify({'error': 'Token requerido'}), 401

            token = token.replace('Bearer ', '')
            user = verify_token(token)
            if not user:
                return jsonify({'error': 'Token inválido'}), 401

            # Agregar usuario a la request
            request.current_user = {
                'id_usuario': user[0],
                'nombre_usuario': user[1],
                'correo': user[2],
                'nombre_completo': user[3]
            }
            return f(*args, **kwargs)

        wrapper.__name__ = f.__name__
        return wrapper

    return decorator


# Crear un blueprint en lugar de una aplicación
blueprint = Blueprint('auth', __name__)

# Configuración de OAuth con Google
def init_oauth(app):
    oauth.init_app(app)
    return oauth.register(
        name='google',
        client_id=os.environ.get("GOOGLE_CLIENT_ID"),
        client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
        access_token_url='https://oauth2.googleapis.com/token',
        access_token_params=None,
        authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
        authorize_params=None,
        api_base_url='https://www.googleapis.com/oauth2/v3/',
        userinfo_endpoint='https://openidconnect.googleapis.com/v1/userinfo',
        jwks_uri='https://www.googleapis.com/oauth2/v3/certs',
        client_kwargs={'scope': 'openid email profile'},
    )
# ENDPOINTS DE AUTENTICACIÓN

@blueprint.route('/login')
@cross_origin()
def login():
    """Inicia el proceso de autenticación con Google"""
    redirect_uri = url_for('auth.auth_callback', _external=True)
    return google.authorize_redirect(redirect_uri)


@blueprint.route('/login/credentials', methods=['POST'])
@cross_origin()
def login_credentials():
    """Autenticación con email y contraseña"""
    try:
        # Obtener datos del request
        data = request.get_json()
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({'error': 'Email y contraseña son requeridos'}), 400

        email = data['email']
        password = data['password']

        # Buscar usuario por email
        user_result = db.execute_query(
            """SELECT id_usuario, nombre_usuario, correo, contrasena_hash, 
                      nombre_completo, token, creado_en, actualizado_en 
               FROM usuarios WHERE correo = %s""",
            (email,)
        )

        if not user_result:
            return jsonify({'error': 'Credenciales inválidas'}), 401

        user_data = user_result[0]
        stored_password_hash = user_data[3]

        # Verificar contraseña (comparación directa)
        if stored_password_hash != password:
            return jsonify({'error': 'Credenciales inválidas'}), 401

        # Generar nuevo token
        new_token = generate_token()

        # Actualizar token en la base de datos
        db.execute_query(
            "UPDATE usuarios SET token = %s, actualizado_en = CURRENT_TIMESTAMP WHERE id_usuario = %s",
            (new_token, user_data[0]),
            fetch=False
        )

        # Retornar datos del usuario
        return jsonify({
            'success': True,
            'user': {
                'id_usuario': user_data[0],
                'nombre_usuario': user_data[1],
                'correo': user_data[2],
                'nombre_completo': user_data[4],
                'token': new_token,
                'creado_en': user_data[6].isoformat() if user_data[6] else None,
                'actualizado_en': user_data[7].isoformat() if user_data[7] else None
            }
        }), 200

    except Exception as e:
        print(f"Error en login_credentials: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@blueprint.route('/auth/callback')
@cross_origin()
def auth_callback():
    """Callback de Google OAuth"""
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')

        if user_info:
            # Crear o actualizar usuario
            user_data = create_or_update_user(user_info)

            # Guardar en sesión
            session['user_token'] = user_data['token']
            session['user_id'] = user_data['id_usuario']

            return redirect('/dashboard')
        else:
            return jsonify({'error': 'No se pudo obtener información del usuario'}), 400

    except Exception as e:
        print(f"Error en auth_callback: {e}")
        return jsonify({'error': 'Error en el proceso de autenticación'}), 500


@blueprint.route('/dashboard')
@cross_origin()
def dashboard():
    """Dashboard principal (requiere autenticación)"""
    if 'user_token' not in session:
        return redirect('/login')

    try:
        user = verify_token(session['user_token'])
        if not user:
            session.clear()
            return redirect('/login')

        return jsonify({
            'message': 'Bienvenido al dashboard',
            'user': {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'name': user[3]
            }
        })
    except Exception as e:
        print(f"Error en dashboard: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@blueprint.route('/logout')
@cross_origin()
def logout():
    """Cerrar sesión"""
    session.clear()
    return jsonify({'message': 'Sesión cerrada exitosamente'})


# ENDPOINTS DE CURSOS

@blueprint.route('/api/cursos', methods=['GET'])
@require_auth()
@cross_origin()
def get_cursos():
    """Obtiene la lista de cursos del usuario autenticado"""
    try:
        user_id = request.current_user['id_usuario']

        cursos = db.execute_query(
            """SELECT c.id_curso, c.titulo, c.descripcion, c.nivel_profundidad, 
                      c.duracion_total, c.creado_en, c.estado, c.es_publico, c.idioma,
                      COUNT(sc.id_seccion) as total_secciones
               FROM cursos c
               LEFT JOIN secciones_curso sc ON c.id_curso = sc.id_curso
               WHERE c.id_usuario = %s
               GROUP BY c.id_curso, c.titulo, c.descripcion, c.nivel_profundidad, 
                        c.duracion_total, c.creado_en, c.estado, c.es_publico, c.idioma
               ORDER BY c.creado_en DESC""",
            (user_id,)
        )

        cursos_list = []
        for curso in cursos:
            cursos_list.append({
                'id_curso': curso[0],
                'titulo': curso[1],
                'descripcion': curso[2],
                'nivel_profundidad': curso[3],
                'duracion_total': str(curso[4]) if curso[4] else None,
                'creado_en': curso[5].isoformat() if curso[5] else None,
                'estado': curso[6],
                'es_publico': curso[7],
                'idioma': curso[8],
                'total_secciones': curso[9]
            })

        return jsonify({
            'cursos': cursos_list,
            'total': len(cursos_list)
        })

    except Exception as e:
        print(f"Error en get_cursos: {e}")
        return jsonify({'error': 'Error al obtener cursos'}), 500


@blueprint.route('/api/cursos/<int:curso_id>', methods=['GET'])
@require_auth()
@cross_origin()
def get_curso_detalle(curso_id):
    """Obtiene el detalle completo de un curso específico"""
    try:
        user_id = request.current_user['id_usuario']

        # Verificar que el curso pertenece al usuario
        curso = db.execute_query(
            """SELECT c.id_curso, c.titulo, c.descripcion, c.nivel_profundidad, 
                      c.duracion_total, c.creado_en, c.actualizado_en, c.estado, 
                      c.es_publico, c.idioma
               FROM cursos c
               WHERE c.id_curso = %s AND c.id_usuario = %s""",
            (curso_id, user_id)
        )

        if not curso:
            return jsonify({'error': 'Curso no encontrado'}), 404

        curso_data = curso[0]

        # Obtener secciones del curso
        secciones = db.execute_query(
            """SELECT sc.id_seccion, sc.titulo, sc.descripcion, sc.indice_orden,
                      sc.creado_en, sc.id_seccion_padre
               FROM secciones_curso sc
               WHERE sc.id_curso = %s
               ORDER BY sc.indice_orden""",
            (curso_id,)
        )

        # Obtener videos por sección
        secciones_list = []
        for seccion in secciones:
            seccion_id = seccion[0]

            videos = db.execute_query(
                """SELECT v.id_video, v.id_video_youtube, v.titulo, v.nombre_canal,
                          v.url, v.duracion, v.publicado_en, v.conteo_vistas,
                          v.licencia, vsc.indice_orden
                   FROM videos v
                   JOIN videos_secciones_curso vsc ON v.id_video = vsc.id_video
                   WHERE vsc.id_seccion = %s
                   ORDER BY vsc.indice_orden""",
                (seccion_id,)
            )

            videos_list = []
            for video in videos:
                videos_list.append({
                    'id_video': video[0],
                    'id_video_youtube': video[1],
                    'titulo': video[2],
                    'nombre_canal': video[3],
                    'url': video[4],
                    'duracion': str(video[5]) if video[5] else None,
                    'publicado_en': video[6].isoformat() if video[6] else None,
                    'conteo_vistas': video[7],
                    'licencia': video[8],
                    'orden': video[9]
                })

            secciones_list.append({
                'id_seccion': seccion[0],
                'titulo': seccion[1],
                'descripcion': seccion[2],
                'indice_orden': seccion[3],
                'creado_en': seccion[4].isoformat() if seccion[4] else None,
                'id_seccion_padre': seccion[5],
                'videos': videos_list
            })

        # Obtener etiquetas del curso
        etiquetas = db.execute_query(
            """SELECT e.id_etiqueta, e.nombre
               FROM etiquetas e
               JOIN etiquetas_curso ec ON e.id_etiqueta = ec.id_etiqueta
               WHERE ec.id_curso = %s""",
            (curso_id,)
        )

        etiquetas_list = [{'id_etiqueta': et[0], 'nombre': et[1]} for et in etiquetas]

        curso_completo = {
            'id_curso': curso_data[0],
            'titulo': curso_data[1],
            'descripcion': curso_data[2],
            'nivel_profundidad': curso_data[3],
            'duracion_total': str(curso_data[4]) if curso_data[4] else None,
            'creado_en': curso_data[5].isoformat() if curso_data[5] else None,
            'actualizado_en': curso_data[6].isoformat() if curso_data[6] else None,
            'estado': curso_data[7],
            'es_publico': curso_data[8],
            'idioma': curso_data[9],
            'secciones': secciones_list,
            'etiquetas': etiquetas_list
        }

        return jsonify(curso_completo)

    except Exception as e:
        print(f"Error en get_curso_detalle: {e}")
        return jsonify({'error': 'Error al obtener detalle del curso'}), 500


@blueprint.route('/api/cursos', methods=['POST'])
@require_auth()
@cross_origin()
def crear_curso():
    """Crea un nuevo curso con todas sus dependencias"""
    try:
        user_id = request.current_user['id_usuario']
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Datos requeridos'}), 400

        # Extraer datos del fullData
        full_data = data.get('fullData', {})
        if not full_data:
            return jsonify({'error': 'fullData requerido'}), 400

        # Determinar nivel numérico basado en el nivel de texto
        nivel_map = {
            'principiante': 1,
            'intermedio': 2,
            'avanzado': 3
        }
        nivel_profundidad = nivel_map.get(full_data.get('level', 'principiante'), 1)

        # Calcular duración total si hay secciones
        duracion_total = full_data.get('totalDuration', '0m')

        # Crear el curso principal
        curso_result = db.execute_query(
            """INSERT INTO cursos (
                id_usuario, titulo, descripcion, nivel_profundidad, 
                duracion_total, estado, es_publico, idioma
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id_curso""",
            (
                user_id,
                full_data.get('title', ''),
                full_data.get('introduction', ''),
                nivel_profundidad,
                f"{duracion_total} minutes" if duracion_total != 'N/A' else None,
                'borrador',  # estado
                False,  # es_publico
                full_data.get('language', 'es')
            ),
            fetch=True
        )

        curso_id = curso_result[0][0]

        # Procesar secciones si existen
        if 'sections' in full_data and full_data['sections']:
            for seccion_data in full_data['sections']:
                # Crear sección
                seccion_result = db.execute_query(
                    """INSERT INTO secciones_curso (id_curso, titulo, descripcion, indice_orden)
                       VALUES (%s, %s, %s, %s) RETURNING id_seccion""",
                    (
                        curso_id,
                        seccion_data.get('title', ''),
                        seccion_data.get('content', ''),
                        seccion_data.get('id', seccion_data.get('indice_orden', 1))
                    ),
                    fetch=True
                )

                seccion_id = seccion_result[0][0]

                # Si hay video asociado, crearlo
                if seccion_data.get('videoId') and seccion_data.get('videoTitle'):
                    # Primero verificar si el video ya existe
                    video_existente = db.execute_query(
                        "SELECT id_video FROM videos WHERE id_video_youtube = %s",
                        (seccion_data['videoId'],)
                    )

                    if video_existente:
                        video_id = video_existente[0][0]
                    else:
                        # Crear nuevo video
                        duracion_video = seccion_data.get('duration', 'N/A')
                        if duracion_video == 'N/A':
                            duracion_interval = None
                        else:
                            # Convertir formato "X min" a interval
                            try:
                                minutes = int(duracion_video.replace(' min', '').replace('m', ''))
                                duracion_interval = f"{minutes} minutes"
                            except:
                                duracion_interval = None

                        video_result = db.execute_query(
                            """INSERT INTO videos (id_video_youtube, titulo, nombre_canal, url, 
                                                   duracion, licencia, conteo_vistas)
                               VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id_video""",
                            (
                                seccion_data['videoId'],
                                seccion_data['videoTitle'],
                                'Canal YouTube',  # Valor por defecto
                                seccion_data.get('videoUrl',
                                                 f"https://www.youtube.com/watch?v={seccion_data['videoId']}"),
                                duracion_interval,
                                'Standard YouTube License',  # Valor por defecto
                                0  # Conteo inicial
                            ),
                            fetch=True
                        )
                        video_id = video_result[0][0]

                    # Asociar video con sección
                    db.execute_query(
                        """INSERT INTO videos_secciones_curso (id_seccion, id_video, indice_orden)
                           VALUES (%s, %s, %s)""",
                        (seccion_id, video_id, 1),
                        fetch=False
                    )

        # Crear etiquetas por defecto basadas en el título y nivel
        etiquetas_default = [
            full_data.get('level', 'principiante'),
            full_data.get('language', 'español')
        ]

        for etiqueta_nombre in etiquetas_default:
            # Verificar si la etiqueta existe
            etiqueta_existente = db.execute_query(
                "SELECT id_etiqueta FROM etiquetas WHERE nombre = %s",
                (etiqueta_nombre,)
            )

            if etiqueta_existente:
                etiqueta_id = etiqueta_existente[0][0]
            else:
                # Crear nueva etiqueta
                etiqueta_result = db.execute_query(
                    "INSERT INTO etiquetas (nombre) VALUES (%s) RETURNING id_etiqueta",
                    (etiqueta_nombre,),
                    fetch=True
                )
                etiqueta_id = etiqueta_result[0][0]

            # Asociar etiqueta con curso
            try:
                db.execute_query(
                    "INSERT INTO etiquetas_curso (id_curso, id_etiqueta) VALUES (%s, %s)",
                    (curso_id, etiqueta_id),
                    fetch=False
                )
            except:
                # Si ya existe la asociación, ignorar el error
                pass

        return jsonify({
            'message': 'Curso creado exitosamente',
            'id_curso': curso_id,
            'titulo': full_data.get('title', '')
        }), 201

    except Exception as e:
        print(f"Error en crear_curso: {e}")
        return jsonify({'error': 'Error al crear el curso'}), 500


# Endpoint adicional para obtener información del usuario autenticado
@blueprint.route('/api/user/me', methods=['GET'])
@require_auth()
@cross_origin()
def get_current_user():
    """Obtiene información del usuario autenticado"""
    return jsonify({
        'user': request.current_user
    })

# No ejecutar la aplicación aquí, ya que será importada como blueprint
if __name__ == "__main__":
    print("Este archivo debe ser importado como blueprint")