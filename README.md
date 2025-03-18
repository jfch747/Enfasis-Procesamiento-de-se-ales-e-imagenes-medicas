# Enfasis-Procesamiento-de-señales-e-imagenes-medicas

# Detección de Manos y Extracción de Coordenadas con MediaPipe
- manos.py
Este repositorio contiene un script en Python que utiliza la biblioteca MediaPipe para detectar manos en videos y extraer las coordenadas de los puntos clave (landmarks) del dedo índice. Las coordenadas se guardan en archivos CSV para su posterior análisis.

Requisitos
Python 3.x

OpenCV (cv2)

MediaPipe (mediapipe)

CSV (biblioteca estándar de Python)

Instalación
Clona el repositorio:

bash
git clone https://github.com/tuusuario/manos-detector.git
cd manos-detector
Instala las dependencias:

bash
pip install opencv-python mediapipe
Uso
Preparación de los videos: Coloca los videos que deseas procesar en una carpeta. Asegúrate de que los videos estén en formato .mp4.

Configuración de rutas: Modifica las variables video_folder y output_folder en el script manos.py para especificar la carpeta donde se encuentran los videos y la carpeta donde se guardarán los archivos CSV con las coordenadas.

Ejecución del script: Ejecuta el script manos.py:

bash
python manos.py
Resultados: El script procesará cada video, detectará las manos y guardará las coordenadas de los puntos clave del dedo índice en archivos CSV en la carpeta de salida especificada.

Estructura del Código
Detección de Manos: Se utiliza mediapipe.solutions.hands para detectar las manos en cada frame del video.

Extracción de Coordenadas: Se extraen las coordenadas de los puntos clave del dedo índice (MCP, PIP y TIP) y se almacenan en listas.

Visualización: Se muestra el video con los landmarks de las manos dibujados en tiempo real.

Guardado de Datos: Las coordenadas se guardan en archivos CSV, uno por cada video procesado.

Parámetros Ajustables
window_width y window_height: Tamaño de la ventana donde se muestra el video.

video_folder: Carpeta que contiene los videos a procesar.

output_folder: Carpeta donde se guardarán los archivos CSV con las coordenadas.

min_detection_confidence y min_tracking_confidence: Umbrales de confianza para la detección y seguimiento de manos.

Ejemplo de Archivo CSV
Cada archivo CSV generado tendrá el siguiente formato:

Index Finger MCP X	Index Finger MCP Y	Index Finger PIP X	Index Finger PIP Y	Index Finger TIP X	Index Finger TIP Y
320	                240	                310	                230	                300	                220
325	                245	                315	                235	                305	                225
...	                ...	                ...	                ...	                ...	                ...
Contribuciones
Las contribuciones son bienvenidas. Si tienes alguna mejora o sugerencia, por favor abre un issue o envía un pull request.

Licencia
Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.

Nota: Asegúrate de tener los videos en la carpeta correcta y de que los archivos tengan la extensión .mp4. El script está diseñado para procesar videos con una o dos manos visibles.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- Entrenamiento de una Red Neuronal Multicapa (MLP) con PyTorch

Este repositorio contiene un cuaderno de Jupyter (`entrenamiento_nn_mlp.ipynb`) que implementa el entrenamiento de una red neuronal multicapa (MLP) utilizando PyTorch. El modelo se entrena en un conjunto de datos específico y se evalúa utilizando métricas como precisión, recall y F1-score. Además, se utiliza SHAP para explicar las predicciones del modelo.

## Contenido del Repositorio

- **`entrenamiento_nn_mlp.ipynb`**: Cuaderno de Jupyter que contiene el código para el entrenamiento y evaluación de la red neuronal.
- **`csv_f_2.csv`**: Archivo CSV que contiene los datos de entrenamiento utilizados en el cuaderno.

## Requisitos

Para ejecutar el cuaderno, necesitarás las siguientes bibliotecas de Python:

- `torch`
- `torch.nn`
- `torch.optim`
- `matplotlib`
- `numpy`
- `pandas`
- `sklearn`
- `seaborn`
- `shap`

Puedes instalar estas dependencias utilizando `pip`:

bash
pip install torch matplotlib numpy pandas scikit-learn seaborn shap
Estructura del Código
Importación de bibliotecas: Se importan las bibliotecas necesarias para el entrenamiento y evaluación del modelo.

Configuración del dispositivo: Se verifica si hay una GPU disponible y se configura el dispositivo en consecuencia.

Carga de datos: Se carga el archivo CSV csv_f_2.csv y se preparan los datos para el entrenamiento.

Preprocesamiento de datos: Se codifican las etiquetas categóricas y se normalizan las características.

Definición del modelo: Se define la arquitectura de la red neuronal multicapa.

Configuración del entrenamiento: Se establecen los hiperparámetros del modelo.

Bucle de validación cruzada: Se realiza la validación cruzada para entrenar y evaluar el modelo.

Entrenamiento del modelo: Se entrena el modelo utilizando el optimizador Adam y la función de pérdida de entropía cruzada.

Visualización de resultados: Se visualizan los resultados del entrenamiento, incluyendo la matriz de confusión.

Uso de SHAP para explicabilidad: Se utiliza SHAP para explicar las predicciones del modelo.

Evaluación del modelo: Se evalúa el modelo en el conjunto de prueba y se calculan métricas adicionales.

Guardado del modelo: El modelo entrenado se guarda para su uso posterior.

Ejecución
Para ejecutar el cuaderno, asegúrate de tener instaladas todas las dependencias necesarias. Luego, abre el cuaderno en Jupyter y ejecuta las celdas en orden.

bash
Copy
jupyter notebook entrenamiento_nn_mlp.ipynb
Contribuciones
Si deseas contribuir a este proyecto, por favor sigue los siguientes pasos:

Haz un fork del repositorio.

Crea una nueva rama (git checkout -b feature/nueva-funcionalidad).

Realiza tus cambios y haz commit (git commit -am 'Añade nueva funcionalidad').

Haz push a la rama (git push origin feature/nueva-funcionalidad).

Abre un Pull Request.

Licencia
Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.



--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- Este repositorio contiene un cuaderno de Jupyter (entrenamiento_nn_lstm.ipynb) que implementa una red neuronal LSTM (Long Short-Term Memory) para la clasificación de señas basadas en datos de coordenadas de dedos. El objetivo es entrenar un modelo de red neuronal que pueda predecir la seña correspondiente a partir de las coordenadas de los dedos.

Descripción del Proyecto
El proyecto se centra en el entrenamiento de una red neuronal LSTM utilizando datos de coordenadas de dedos para clasificar diferentes señas. El modelo se entrena utilizando un conjunto de datos que contiene las coordenadas X e Y de los dedos índice (MCP, PIP, TIP) y la seña correspondiente.

Estructura del Proyecto
Carga de Datos: Los datos se cargan desde un archivo CSV que contiene las coordenadas de los dedos y las señas correspondientes.

Preprocesamiento: Los datos se normalizan y se codifican las etiquetas de las señas para su uso en el modelo.

Definición del Modelo: Se define una red neuronal LSTM con una capa oculta y una capa de salida para la clasificación.

Entrenamiento: El modelo se entrena utilizando validación cruzada y se evalúa su rendimiento en términos de precisión, recall y F1-score.

Visualización de Resultados: Se visualizan las métricas de rendimiento y se genera una matriz de confusión para evaluar el modelo.

Requisitos
Para ejecutar este cuaderno, necesitarás las siguientes bibliotecas de Python:

torch

torch.nn

torch.optim

pandas

numpy

sklearn

matplotlib

seaborn

shap

Puedes instalar las dependencias necesarias utilizando el siguiente comando:

bash

pip install torch pandas numpy scikit-learn matplotlib seaborn shap
Uso
Clona el repositorio:

bash

git clone https://github.com/tu_usuario/entrenamiento_nn_lstm.git
cd entrenamiento_nn_lstm
Abre el cuaderno de Jupyter:

bash

jupyter notebook entrenamiento_nn_lstm.ipynb
Ejecuta el cuaderno: Sigue las instrucciones en el cuaderno para cargar los datos, entrenar el modelo y evaluar su rendimiento.

Estructura del Código
Importación de Bibliotecas: Se importan las bibliotecas necesarias para el procesamiento de datos, la definición del modelo y la visualización.

Carga de Datos: Los datos se cargan desde un archivo CSV y se preprocesan para su uso en el modelo.

Definición del Modelo: Se define una red neuronal LSTM con una capa oculta y una capa de salida.

Entrenamiento y Validación: El modelo se entrena utilizando validación cruzada y se evalúa su rendimiento.

Visualización de Resultados: Se generan gráficos y matrices de confusión para evaluar el rendimiento del modelo.

Resultados
El modelo se evalúa utilizando métricas como la precisión, el recall y el F1-score. Además, se genera una matriz de confusión para visualizar el rendimiento del modelo en la clasificación de las diferentes señas.

Contribuciones
Las contribuciones son bienvenidas. Si encuentras algún error o tienes alguna sugerencia para mejorar el modelo, no dudes en abrir un issue o enviar un pull request.

Licencia
Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.
