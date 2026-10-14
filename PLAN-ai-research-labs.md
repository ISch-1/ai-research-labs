# Plan: AI Research Labs — portafolio académico de IA para GitHub

## Contexto
Quieres un proyecto público en GitHub que sirva para el CV y para demostrar conocimientos de IA
de forma verificable. Se divide en 3 laboratorios: (A) ML clásico/estadística/forecasting con API,
(B) NLP y LLMs (RAG + evaluación), (C) Visión por computador. El resultado debe ser reproducible,
con resultados medidos, informes claros y buenas prácticas de ingeniería (tests, CI, Docker).

**Decisiones por defecto** (cambiables antes de empezar):
- **Monorepo** `ai-research-labs` con 3 laboratorios + paquete común → un solo enlace fuerte en el CV.
- **Python 3.12** (igual que `db-sentinel` y `ritual-ai-service`), gestor `uv` (o `pip` + `requirements.txt`), PyTorch para deep learning.
- **Estilo de README** igual al de `ISch-1/db-sentinel`: badges, blockquote de resumen, diagrama ASCII del flujo, tabla "Why it exists" que mapea cada capacidad a un requisito de puesto, Quickstart con `docker compose up`.
- **Hardware:** todo debe correr en CPU con datasets pequeños; entrenamientos pesados en Google Colab/Kaggle (notebook de entrada incluido).
- **LLMs:** modelos locales vía **Ollama** (gratis) + **una API comercial opcional** (p. ej. Claude Haiku/Sonnet) con presupuesto limitado, para comparar coste/latencia/calidad.
- **Documentación en inglés** (alcance internacional), con `README.es.md` opcional.

### Relación con tus repos existentes (revisados)
| Repo actual | Qué contiene | Cómo se aprovecha en este proyecto |
|---|---|---|
| `TPOCLUSTERING` (local) | K-Means K=3 sobre Wholesale customers (UCI), silhouette ≈ 0.548 | **A3**: pasa a ser el *baseline*; se añade DBSCAN/HDBSCAN/GMM, estabilidad por bootstrap y anomalías sobre el mismo dataset → muestra evolución |
| `ISch-1/tp-Ciencia-de-Datos` (privado) | California Housing, comparación de modelos, residuos | **A1**: se migra a pipelines + CV repetida + tests estadísticos (lo que le falta: IC y significancia) |
| `ISch-1/grocery-sales-forecasting` (privado) | Holt-Winters + Random Forest para demanda | **A2**: misma idea sobre datos públicos (los datos de la verdulería no se publican), añadiendo SARIMA, LightGBM con lags, LSTM/N-BEATS y backtesting |
| `ISch-1/db-sentinel` (público) | Isolation Forest para anomalías en métricas de BD | **A3**: se cita como "aplicación en producción" del lab de anomalías; no se duplica |
| `ritual-ai-service` (local) | FastAPI + Redis/RQ, LLM que "solo narra" | **A5/B5**: reutilizar el patrón FastAPI + `/health` + `.env.example`; el Lab B aporta la evaluación rigurosa de LLM que ese servicio no tiene |

Con esto el portafolio cuenta una historia coherente: proyectos académicos → laboratorio riguroso → sistemas en producción (`db-sentinel`, `ritual`). En el README raíz enlazar estos repos en una sección "Related work".

---

## Fase 0 — Base del repositorio (1 semana)

1. **Crear repo** `ai-research-labs` en GitHub con licencia MIT, `.gitignore` de Python, `README.md` raíz.
2. **Estructura:**
   ```
   ai-research-labs/
   ├── common/              # paquete compartido (pip install -e .)
   │   ├── data.py          # descarga/caché de datasets
   │   ├── metrics.py       # métricas + intervalos de confianza (bootstrap)
   │   ├── stats.py         # tests estadísticos (Wilcoxon, Friedman, McNemar)
   │   ├── tracking.py      # wrapper de MLflow
   │   └── seed.py          # reproducibilidad (semillas)
   ├── lab-a-predictive/
   ├── lab-b-nlp-llm/
   ├── lab-c-vision/
   ├── docs/                # informes y figuras
   ├── tests/
   ├── pyproject.toml
   ├── Makefile             # make setup / test / lint / train-a / serve-a ...
   ├── docker-compose.yml
   └── .github/workflows/ci.yml
   ```
   Cada lab: `configs/` (YAML), `src/`, `notebooks/` (solo exploración), `experiments/` (scripts), `reports/`, `README.md`.
3. **Calidad de código:** `ruff` (lint+format), `mypy` básico, `pytest`, `pre-commit`.
4. **Tracking de experimentos:** MLflow local (`mlruns/` ignorado en git); cada experimento registra config, métricas, semilla y artefactos.
5. **Configuración:** Hydra o YAML + `pydantic` para que cada experimento sea `python -m lab_a.run config=xxx.yaml`.
6. **CI (GitHub Actions):** lint + tests en cada push con datasets *mini* (smoke tests < 5 min).
7. **Convención de commits** (Conventional Commits) y ramas por feature; issues/milestones por fase → muestra gestión de proyecto.

**Entregable:** repo vacío pero profesional, CI en verde, `make setup` funcionando.

---

## Fase 1 — Lab A: Predictive & Statistical Modeling Lab (3–4 semanas)

### A1. Regresión y clasificación
1. **Datasets:** California Housing (regresión), Adult Income o Bank Marketing (clasificación tabular desbalanceada), Wine Quality.
2. **EDA reproducible:** script que genera perfil de datos (nulos, distribuciones, correlaciones) → `reports/eda_*.md` con figuras.
3. **Pipelines `scikit-learn`:** imputación, escalado, one-hot, todo dentro de `Pipeline`/`ColumnTransformer` (evita *data leakage*).
4. **Modelos a comparar:**
   - Estadísticos: regresión lineal/Ridge/Lasso, regresión logística (con interpretación de coeficientes con `statsmodels`).
   - Clásicos: árbol, Random Forest, Gradient Boosting (XGBoost/LightGBM), SVM, KNN.
   - Red neuronal: MLP en PyTorch (o `MLPClassifier`) con early stopping.
5. **Protocolo:** validación cruzada estratificada repetida (5×2 o 10-fold), búsqueda de hiperparámetros con Optuna, test set final intocable.
6. **Métricas:** RMSE/MAE/R² ; Accuracy, F1, ROC-AUC, PR-AUC, calibración (Brier, curva de calibración).
7. **Interpretabilidad:** importancia por permutación y SHAP para el mejor modelo.

### A2. Forecasting de series temporales
1. **Datasets:** Air Passengers / M4 subset / consumo eléctrico (UCI) / datos de ventas.
2. **Modelos:** baselines (naive, seasonal naive, media móvil) → ARIMA/SARIMA, ETS (Holt-Winters) → Prophet → LightGBM con features de lags → LSTM/GRU o N-BEATS (via `darts` o PyTorch).
3. **Validación:** backtesting con ventana deslizante (rolling origin), nunca shuffle.
4. **Métricas:** MAE, RMSE, MAPE, sMAPE, MASE; intervalos de predicción y su cobertura.
5. **Análisis:** descomposición STL, ACF/PACF, test de estacionariedad (ADF/KPSS).

### A3. Clustering y anomalías
1. **Datasets:** Mall Customers / Online Retail (segmentación), Credit Card Fraud (anomalías, Kaggle), sintéticos de `make_blobs`.
2. **Clustering:** K-Means, DBSCAN/HDBSCAN, Gaussian Mixture, jerárquico; elección de k con codo, silhouette, Davies-Bouldin; visualización con PCA/UMAP.
3. **Anomalías:** Isolation Forest, LOF, One-Class SVM, z-score/IQR estadístico y **autoencoder** (error de reconstrucción).
4. **Evaluación:** con etiquetas disponibles → PR-AUC, recall@k; sin etiquetas → estabilidad de clusters (bootstrap, ARI entre ejecuciones).

### A4. Comparación estadística de resultados (diferenciador clave)
1. Implementar en `common/stats.py`:
   - Intervalos de confianza por **bootstrap** para cualquier métrica.
   - **Wilcoxon signed-rank** / t-test pareado corregido (Nadeau-Bengio) entre 2 modelos en CV.
   - **Friedman + post-hoc Nemenyi** para varios modelos en varios datasets, con **critical difference diagram**.
   - **McNemar** para clasificadores en el mismo test set.
   - **Diebold-Mariano** para forecasting.
2. Generar tabla final: media ± IC95%, ranking medio, significancia.
3. Informe `docs/lab-a-report.md`: "¿Ganan siempre los modelos complejos?" — conclusión basada en evidencia.

### A5. API para servir modelos
1. **FastAPI** con endpoints `/predict/regression`, `/predict/classification`, `/forecast`, `/anomaly`, `/health`, `/models` (metadatos/versiones).
2. Validación de entrada con **Pydantic**; carga de modelos desde MLflow Model Registry o `joblib`.
3. **Docker** multi-stage + `docker-compose` (API + MLflow).
4. Tests de la API (`pytest` + `httpx`), logging estructurado, métricas de latencia (Prometheus opcional).
5. Opcional: demo con **Streamlit** o despliegue gratuito (Render / Hugging Face Spaces).

**Demuestra:** fundamentos de ML, rigor experimental, estadística y MLOps básico.

---

## Fase 2 — Lab B: NLP & LLM Research Lab (4 semanas)

### B1. TF-IDF frente a embeddings
1. **Datasets:** AG News, 20 Newsgroups, IMDB/SST-2; opcional dataset en español (p. ej. reseñas de MercadoLibre/Amazon ES) → aporta valor diferencial.
2. **Representaciones:** Bag-of-Words, TF-IDF (uni/bi-gramas), Word2Vec/GloVe promediado, sentence-transformers (`all-MiniLM-L6-v2`, `multilingual-e5`), embeddings de API (opcional).
3. **Análisis:** dimensionalidad, tiempo de cómputo, memoria, visualización UMAP coloreada por clase.

### B2. Clasificación de texto
1. Escalera de modelos: TF-IDF + LogReg/SVM → embeddings + LogReg → fine-tuning de **DistilBERT/BERT** (Hugging Face `Trainer`, en Colab) → **zero-shot/few-shot con LLM**.
2. Curvas de aprendizaje: rendimiento vs nº de ejemplos etiquetados (100, 1k, 10k) → ¿cuándo compensa un LLM?
3. Análisis de errores: ejemplos mal clasificados, matriz de confusión, sesgos.
4. Comparación estadística reutilizando `common/stats.py` (McNemar, bootstrap).

### B3. Búsqueda semántica y RAG
1. **Corpus:** documentación técnica pública (p. ej. docs de Python/FastAPI) o papers de arXiv; dataset de evaluación con preguntas-respuestas (crear ~100 pares, algunos generados y revisados manualmente; o usar SQuAD/Natural Questions subset).
2. **Indexación:** chunking (tamaño fijo vs por secciones vs semántico), embeddings, vector store (**FAISS** o **Chroma**/Qdrant).
3. **Recuperación:** BM25 vs densa vs **híbrida** (fusión RRF) + **re-ranking** con cross-encoder.
4. **Métricas de recuperación:** Recall@k, MRR, nDCG.
5. **Generación:** pipeline RAG propio (sin depender en exceso de frameworks; LangChain/LlamaIndex solo como comparación), con citas a las fuentes.

### B4. Evaluación de respuestas de LLMs
1. Métricas de referencia: Exact Match, F1 de tokens, BERTScore.
2. **LLM-as-a-judge** con rúbrica (fidelidad/faithfulness, relevancia, completitud) y verificación de concordancia con anotación humana en una muestra (Cohen's kappa).
3. Métricas RAG estilo **RAGAS**: faithfulness, answer relevancy, context precision/recall.
4. Pruebas de robustez: preguntas sin respuesta en el corpus (¿alucina o dice "no sé"?), prompt injection en documentos.

### B5. Comparación de coste, latencia y calidad
1. Modelos: 2–3 locales vía Ollama (p. ej. Llama 3.x 3B/8B, Qwen, Mistral) + 1–2 de API (modelo pequeño vs grande).
2. Medir por consulta: tokens de entrada/salida, coste estimado (USD), latencia p50/p95, time-to-first-token, calidad (B4).
3. Gráfica **frontera de Pareto** calidad vs coste y calidad vs latencia.
4. Experimentos: efecto del nº de chunks recuperados, del tamaño de chunk y de la temperatura.
5. API: endpoint `/ask` en FastAPI + UI Streamlit/Gradio para demo del RAG.
6. Gestión segura de claves: `.env` + `.env.example`, nunca claves en el repo; caché de respuestas para no repetir gasto.

**Demuestra:** IA generativa actual, recuperación de información y evaluación rigurosa de sistemas con LLMs.

---

## Fase 3 — Lab C: Computer Vision Research Lab (3–4 semanas)

### C1. Perceptrón y redes neuronales
1. **Perceptrón implementado desde cero en NumPy** (incluido el algoritmo de aprendizaje) sobre datos linealmente separables y el caso XOR (muestra por qué falla).
2. **MLP desde cero** (forward + backprop manual) sobre MNIST → luego la misma red en PyTorch; verificar que los gradientes coinciden (gradient check).
3. Experimentos: funciones de activación, inicialización, learning rate, optimizadores (SGD, Momentum, Adam), batch norm, dropout.

### C2. CNN y transfer learning
1. **Datasets:** MNIST/Fashion-MNIST (base), CIFAR-10, y uno "real" pequeño (Oxford Flowers-102, Oxford Pets o un dataset médico tipo chest X-ray de Kaggle).
2. CNN propia (LeNet → VGG-like pequeña) entrenada desde cero.
3. **Transfer learning** con `torchvision`/`timm`: ResNet-18/50, EfficientNet, MobileNet, ViT pequeño; comparar *feature extraction* vs *fine-tuning* completo.
4. Comparar: precisión, nº de parámetros, tiempo de entrenamiento, latencia de inferencia en CPU.

### C3. Data augmentation
1. Ablación: sin augmentation vs flips/crops vs color jitter vs RandAugment/AutoAugment vs Mixup/CutMix.
2. Experimento de **pocos datos**: entrenar con 10%, 25%, 50%, 100% del dataset con y sin augmentation.
3. Visualizar las transformaciones aplicadas (grid de imágenes).

### C4. Matrices de confusión y análisis de errores
1. Matriz de confusión normalizada, métricas por clase, top-k accuracy.
2. Galería de **errores más confiados** (alta confianza, predicción incorrecta) y pares de clases más confundidos.
3. Calibración (ECE, reliability diagram) y *temperature scaling*.
4. Robustez: rendimiento con ruido, desenfoque y rotación (estilo CIFAR-10-C).

### C5. Interpretabilidad visual
1. **Grad-CAM / Grad-CAM++**, saliency maps, Integrated Gradients (librerías `captum` o `pytorch-grad-cam`).
2. Visualización de filtros y mapas de activación de las primeras capas.
3. Caso de estudio: detectar si el modelo usa "atajos" (fondos, marcas de agua) mirando los mapas de atención.
4. Embeddings de imágenes con t-SNE/UMAP.
5. Demo: Gradio que clasifica una imagen subida y muestra el Grad-CAM; endpoint `/classify-image` en la API.

**Demuestra:** deep learning, experimentación con datos no estructurados e interpretabilidad.

---

## Fase 4 — Pulido para CV (1–2 semanas)
1. **README raíz** con: resumen, diagrama de arquitectura, tabla de resultados principales de cada lab, badges (CI, licencia, Python), GIFs de las demos, cómo reproducir en 3 comandos.
2. **Informe por lab** en `docs/` (estilo mini-paper: pregunta, método, resultados con IC, conclusiones, limitaciones).
3. **Model cards** y **data cards** para modelos y datasets usados (ética, sesgos, licencias).
4. Demos desplegadas (Hugging Face Spaces gratis) y enlace en el README.
5. Post en LinkedIn / blog técnico resumiendo un hallazgo concreto por lab.
6. Releases etiquetadas (`v0.1-lab-a`, `v0.2-lab-b`, `v1.0`) → historial visible de progreso.

---

## Stack tecnológico resumido
| Área | Herramientas |
|---|---|
| Datos/ML | pandas, numpy, scikit-learn, statsmodels, XGBoost/LightGBM, Optuna, SHAP |
| Series temporales | statsmodels, Prophet, darts |
| Deep Learning | PyTorch, torchvision, timm, captum, pytorch-grad-cam |
| NLP/LLM | Hugging Face transformers, sentence-transformers, rank-bm25, FAISS/Chroma, Ollama, SDK de la API elegida, RAGAS |
| Estadística | scipy.stats, scikit-posthocs, autorank |
| MLOps | MLflow, FastAPI, Pydantic, Docker, GitHub Actions, pytest, ruff, pre-commit |
| Demos | Streamlit / Gradio, Hugging Face Spaces |

## Cronograma orientativo (~14 semanas a tiempo parcial)
| Semana | Trabajo |
|---|---|
| 1 | Fase 0: repo, estructura, CI, `common/` |
| 2–5 | Lab A (A1→A5) |
| 6–9 | Lab B (B1→B5) |
| 10–13 | Lab C (C1→C5) |
| 14 | Fase 4: pulido, informes, demos, publicación |

## Archivos clave a crear
- `common/stats.py`, `common/metrics.py` — reutilizados por los 3 labs (comparación estadística única).
- `lab-a-predictive/src/api/main.py` — API FastAPI (luego ampliada con `/ask` y `/classify-image`).
- `lab-b-nlp-llm/src/rag/` (`chunking.py`, `retriever.py`, `generator.py`, `evaluate.py`).
- `lab-c-vision/src/` (`perceptron_numpy.py`, `mlp_numpy.py`, `train.py`, `interpret.py`).
- `docs/lab-{a,b,c}-report.md`, `README.md`, `.github/workflows/ci.yml`, `docker-compose.yml`.

## Verificación
- `make test`: tests unitarios de `common/` (p. ej. bootstrap da IC correcto en datos sintéticos; Friedman coincide con `scipy`), gradient check del MLP NumPy, tests de la API con `httpx`.
- `make smoke`: cada experimento corre con un dataset mini en CPU en < 5 min (también en CI).
- `docker compose up` → `curl localhost:8000/health` y una predicción de cada endpoint.
- Reproducibilidad: dos ejecuciones con la misma semilla dan las mismas métricas (± tolerancia GPU).
- Revisión final: un tercero clona el repo y reproduce un resultado del README siguiendo solo las instrucciones.

## Siguiente paso tras aprobar
Crear la carpeta del proyecto (elegida por ti) e implementar la **Fase 0** completa como primera entrega.
