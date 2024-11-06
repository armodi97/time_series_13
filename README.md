
# Proyecto de Predicción de Pedidos de Taxi en Horas Pico

Este proyecto desarrolla un modelo de machine learning para predecir el número de pedidos de taxi en la próxima hora en aeropuertos. La empresa Sweet Lift Taxi busca una herramienta para optimizar la disponibilidad de conductores en horas pico.

## Descripción del Proyecto

Sweet Lift Taxi desea implementar un modelo que permita anticipar la demanda de taxis en una ventana de tiempo de una hora, basándose en datos históricos de pedidos en aeropuertos. Para ser efectivo, el modelo debe cumplir con una métrica de **RMSE** no superior a 48 en el conjunto de prueba.

## Estructura del Análisis

1. **Preparación de los Datos**
   - Se cargan y preparan los datos del archivo `taxi.csv`.
   - Los datos se remuestrean por hora para estandarizar la frecuencia de la serie temporal.

2. **Análisis Exploratorio de Datos (EDA)**
   - Se exploran las características de la serie temporal y se examinan patrones de estacionalidad y tendencia utilizando descomposición estacional.
   
3. **Entrenamiento de Modelos**
   - Se prueban diferentes modelos de regresión, incluyendo `LinearRegression`, `RandomForestRegressor`, `SARIMAX`, `LightGBM`, y `CatBoostRegressor`.
   - Se dividen los datos en entrenamiento y prueba, asignando el 10% de los datos al conjunto de prueba.

4. **Evaluación de Modelos**
   - Los modelos se evalúan utilizando la métrica **Root Mean Squared Error (RMSE)**.
   - Se selecciona el modelo que mejor predice la demanda manteniendo un RMSE por debajo de 48.

5. **Conclusiones**
   - Se concluye con la selección del modelo óptimo para la predicción de pedidos de taxi y recomendaciones para su implementación en la aplicación de Sweet Lift Taxi.

## Requisitos

- **Python 3.7+**
- Librerías: `pandas`, `numpy`, `plotly`, `matplotlib`, `statsmodels`, `scikit-learn`, `lightgbm`, `catboost`

## Cómo Ejecutar el Proyecto

1. Clona el repositorio.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta el notebook `proyect_sprint_13.ipynb` para reproducir el análisis y las visualizaciones.
