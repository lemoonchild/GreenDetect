# 🌿 GreenDetect - Sistema Inteligente de Detección de Patologías en Plantas

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**Sistema de Deep Learning para la detección y clasificación automática de enfermedades en plantas usando CNN y Transfer Learning con Explainable AI (XAI)**

[Características](#características) • [Instalación](#instalación) • [Uso](#uso) • [Modelos](#modelos) • [Resultados](#resultados) • [Licencia](#licencia)

</div>

---

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Características Principales](#características-principales)
- [Dataset](#dataset)
- [Modelos Implementados](#modelos-implementados)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Uso](#uso)
- [Resultados](#resultados)
- [Explainable AI (XAI)](#explainable-ai-xai)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Autores](#autores)
- [Licencia](#licencia)

---

## Descripción del Proyecto

**GreenDetect** es un sistema de visión por computadora basado en Deep Learning que permite la **detección automática y clasificación de patologías en plantas**. El sistema identifica 5 tipos diferentes de condiciones en las hojas:

1. **Bacteria** - Infecciones bacterianas
2. **Fungi** - Enfermedades fúngicas
3. **Healthy** - Plantas sanas
4. **Pests** - Plagas
5. **Virus** - Infecciones virales

El proyecto implementa **dos arquitecturas de redes neuronales**:
- **CNN tradicional desde cero** (baseline)
- **ConvNeXt-Large con Transfer Learning** (modelo avanzado)

Además, incorpora técnicas de **Explainable AI (XAI)** usando **Grad-CAM++** para visualizar qué regiones de la imagen influyeron en la decisión del modelo.

---

## Características Principales

### Detección Avanzada
- Clasificación multi-clase de 5 categorías de patologías
- Precisión superior al 93% con ConvNeXt-Large
- Procesamiento de imágenes de 256x256 píxeles
- Inferencia en tiempo real

### Explainable AI (XAI)
- **Grad-CAM++**: Mapas de calor que muestran regiones de interés
- Visualizaciones interpretables para validación médica/agrícola
- Identificación de características relevantes en las hojas

### Transfer Learning
- Uso de **ConvNeXt-Large** pre-entrenado en ImageNet
- Fine-tuning en 2 fases para máximo rendimiento
- Entrenamiento eficiente con menos datos

### Data Augmentation
- Rotaciones, flips, zoom, variaciones de brillo
- Prevención de overfitting
- Aumento artificial del dataset

---

## Dataset

**Fuente**: [Pathogen Dataset - Kaggle](https://www.kaggle.com/datasets/kanishk3813/pathogen-dataset)

### Distribución del Dataset

| Clase     | Cantidad de Imágenes |
|-----------|---------------------|
| Bacteria  | 7,999               |
| Fungi     | 8,000               |
| Healthy   | 8,000               |
| Pests     | 7,999               |
| Virus     | 8,000               |
| **Total** | **39,998**          |

### División de Datos
- **Entrenamiento**: 80% (~31,998 imágenes)
- **Validación**: 20% (~8,000 imágenes)

### Características de las Imágenes
- **Formato**: JPG/PNG
- **Tamaño de entrada**: 256x256 píxeles
- **Canales**: RGB (3 canales)
- **Normalización**: Valores entre 0 y 1

---

## Modelos Implementados

### 1. CNN Tradicional (Baseline)

Arquitectura de red neuronal convolucional diseñada desde cero.

#### Arquitectura
```
Input (256x256x3)
    ↓
[BLOQUE 1] Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
    ↓
[BLOQUE 2] Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
    ↓
[BLOQUE 3] Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout(0.25)
    ↓
[BLOQUE 4] Conv2D(256) → BatchNorm → Conv2D(256) → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(5, softmax)
```

#### Características Técnicas
- **Parámetros totales**: ~15M
- **Optimizador**: Adam (lr=0.001)
- **Función de pérdida**: Categorical Crossentropy
- **Épocas**: 15
- **Batch size**: 32

#### Resultados CNN
| Métrica    | Valor  |
|------------|--------|
| Accuracy   | 82.92% |
| Precision  | 84.29% |
| Recall     | 81.73% |
| F1-Score   | 82.99% |
| AUC        | 96.04% |

---

### 2. ConvNeXt-Large + Transfer Learning (Modelo Avanzado)

Arquitectura moderna basada en ConvNeXt-Large pre-entrenado en ImageNet con 1.4M imágenes.

#### Arquitectura
```
Input (256x256x3)
    ↓
ConvNeXt Preprocessing
    ↓
ConvNeXt-Large Base Model (ImageNet weights)
    ↓
Global Average Pooling
    ↓
Dropout(0.3) → Dense(512, ReLU) → BatchNorm → Dropout(0.3)
    ↓
Dense(5, softmax)
```

#### Estrategia de Entrenamiento en 2 Fases

##### **Fase 1: Feature Extraction (8 épocas)**
- Base model **congelado** (frozen)
- Solo se entrena el clasificador personalizado
- Learning rate: 1e-4
- Batch size: 32

##### **Fase 2: Fine-Tuning (20 épocas)**
- Se **descongelan** las últimas 50 capas del base model
- Ajuste fino de características de alto nivel
- Learning rate: 1e-5 (10x menor)
- Batch size: 16

#### Características Técnicas
- **Parámetros totales**: ~200M
- **Parámetros entrenables (Fase 1)**: ~2.6M
- **Parámetros entrenables (Fase 2)**: ~50M
- **Optimizador**: AdamW
- **Mixed Precision**: FP16 para eficiencia en GPU
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#### Resultados ConvNeXt-Large
| Métrica    | Valor  | Mejora vs CNN |
|------------|--------|---------------|
| Accuracy   | 93.50% | **+12.8%**    |
| Precision  | 94.12% | **+11.7%**    |
| Recall     | 93.28% | **+14.1%**    |
| F1-Score   | 93.69% | **+12.9%**    |
| AUC        | 98.75% | **+2.8%**     |

---

## Instalación

### Requisitos Previos
- Python 3.8 o superior
- GPU con CUDA (recomendado para entrenamiento)
- 8GB de RAM mínimo (16GB recomendado)

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone https://github.com/lemoonchild/GreenDetect.git

# Instalar librerías principales
pip install tensorflow==2.14.0
pip install numpy pandas matplotlib seaborn
pip install opencv-python Pillow
pip install scikit-learn scikit-image

# Para Explainable AI
pip install tf-keras-vis lime grad-cam

# Para descargar el dataset (Kaggle)
pip install kaggle
```

### Configuración de Kaggle

1. Descarga tu archivo `kaggle.json` desde [Kaggle Account Settings](https://www.kaggle.com/settings/account)

2. Configura las credenciales:
```python
import os
os.makedirs('/root/.kaggle', exist_ok=True)

with open('/root/.kaggle/kaggle.json', 'w') as f:
    f.write('{"username":"tu_usuario","key":"tu_api_key"}')

!chmod 600 /root/.kaggle/kaggle.json
```

3. Descarga el dataset:
```bash
kaggle datasets download -d kanishk3813/pathogen-dataset
unzip pathogen-dataset.zip -d pathogen_data
```

---

## 🔍 Explainable AI (XAI)

### ¿Qué es Grad-CAM++?

**Grad-CAM++** (Gradient-weighted Class Activation Mapping Plus Plus) es una técnica de visualización que genera **mapas de calor** mostrando qué regiones de la imagen fueron más importantes para la decisión del modelo.

### ¿Cómo funciona?

1. El modelo hace una predicción
2. Se calculan los gradientes de la clase predicha respecto a la última capa convolucional
3. Se genera un mapa de activación ponderado
4. Se superpone sobre la imagen original como mapa de calor

### Ejemplo de Visualización XAI

Para cada imagen, GreenDetect genera 4 visualizaciones:

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│  Imagen         │  Grad-CAM++     │  Superposición  │  Predicciones   │
│  Original       │  Heatmap        │                 │                 │
│                 │                 │                 │  Top 3:         │
│  [Imagen de     │  [Mapa de       │  [Imagen +      │  1. Fungi: 99%  │
│   hoja con      │   calor rojo/   │   heatmap       │  2. Virus: 0.5% │
│   manchas]      │   azul]         │   combinados]   │  3. Pests: 0.3% │
│                 │                 │                 │                 │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

### Interpretación de los Mapas de Calor

| Color      | Significado                                    |
|------------|------------------------------------------------|
| Rojo    | **Alta importancia** - Región crítica para la decisión |
| Amarillo | **Importancia media** - Contribuye moderadamente |
| Verde   | **Baja importancia** - Influencia mínima      |
| Azul    | **Sin importancia** - No influye en la decisión |

---

## Tecnologías Utilizadas

### Frameworks de Deep Learning
- **TensorFlow 2.14** - Framework principal
- **Keras** - API de alto nivel para redes neuronales
- **Mixed Precision** - Entrenamiento en FP16 para eficiencia

### Arquitecturas
- **ConvNeXt-Large** - Transfer Learning desde ImageNet
- **CNN Custom** - Arquitectura propia desde cero

### Explainable AI
- **Grad-CAM++** - Visualización de activaciones
- **tf-keras-vis** - Librería para XAI en Keras
- **LIME** - Explicaciones locales (opcional)

### Procesamiento de Datos
- **NumPy** - Operaciones numéricas
- **Pandas** - Análisis de datos
- **OpenCV** - Procesamiento de imágenes
- **Pillow** - Carga y manipulación de imágenes
- **scikit-learn** - Métricas y preprocesamiento

### Visualización
- **Matplotlib** - Gráficos y visualizaciones
- **Seaborn** - Visualizaciones estadísticas
- **cv2** - Procesamiento y superposición de imágenes

### Optimización
- **AdamW** - Optimizador con weight decay
- **Mixed Precision Training** - Reducción de memoria y aceleración
- **Data Augmentation** - Aumento artificial del dataset

---

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

<div align="center">

**Desarrollado con 🌿 para mejorar la agricultura mediante IA**

[Volver arriba ⬆️](#-greendetect---sistema-inteligente-de-detección-de-patologías-en-plantas)

</div>