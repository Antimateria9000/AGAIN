# Predictor bursatil con TFT

Este repositorio implementa un pipeline experimental de forecasting bursatil con Temporal Fusion Transformer (TFT), mas una aplicacion Streamlit para inferencia, comparacion historica y benchmark visual.

No es un sistema listo para inversion real. El objetivo actual del repositorio es investigacion aplicada y validacion tecnica del pipeline.

## Alcance actual

- Descarga de datos con `yfinance`
- Ingenieria de features tecnicas
- Entrenamiento TFT con cuantiles
- Inferencia desde la app Streamlit
- Comparacion historica sobre barras reales disponibles
- Benchmark visual con metricas `MAPE`, `MAE`, `RMSE` y `DirAcc`

## Ejecucion

### Entrenamiento

```bash
python start_training.py
```

### Aplicacion Streamlit

```bash
streamlit run app/app.py
```

## Configuracion

Archivo principal:

- `config/config.yaml`

Archivos de tickers:

- `config/tickers_with_names.yaml`
- `config/benchmark_tickers.yaml`

Rutas de artefactos relevantes definidas en config:

- `data/raw_data_path`
- `data/processed_data_path`
- `data/train_processed_df_path`
- `data/val_processed_df_path`
- `paths/models_dir`
- `paths/normalizers_dir`
- `paths/logs_dir`

## Dependencias

Instalacion basica:

```bash
pip install -r requirements.txt
```

Importante:

- `requirements.txt` incluye las dependencias de runtime detectadas en el codigo.
- A dia de hoy no sustituye a un lockfile por plataforma.
- Para reproducibilidad estricta hace falta resolver y congelar el entorno final de Python, Torch, CUDA y PyTorch Forecasting.

## Estructura real del repositorio

```text
app/
  app.py
  benchmark_utils.py
  config_loader.py
  plot_utils.py
config/
  benchmark_tickers.yaml
  config.yaml
  tickers_with_names.yaml
scripts/
  config_manager.py
  runtime_config.py
  data_fetcher.py
  model.py
  prediction_engine.py
  preprocessor.py
  train.py
  debug/
  utils/
    batch_size_estimator.py
    config_validation.py
    data_schema.py
    feature_engineer.py
    model_config.py
    prediction_utils.py
    transfer_weights.py
    validation_utils.py
start_training.py
requirements.txt
README.md
```

## Tests y validacion automatica

El repositorio incluye pruebas estaticas y de coherencia en `tests/` y una CI minima en `.github/workflows/ci.yml`.

Estas comprobaciones cubren:

- coherencia basica de `config.yaml`
- sincronizacion minima del `README`
- presencia de dependencias criticas en `requirements.txt`
- compilacion de los modulos Python mas importantes

## Politica de artefactos

Los siguientes ficheros y carpetas se consideran artefactos derivados y no fuente de verdad del repositorio:

- `data/`
- `logs/`
- `lightning_logs/`
- checkpoints locales en `models/`

Estos artefactos deben regenerarse desde el codigo, la configuracion y los datos de entrada. La fuente de verdad del pipeline esta en `config/`, `scripts/`, `app/`, `start_training.py` y `requirements.txt`.

## Reproducibilidad de resultados

El README no debe interpretarse como evidencia de rendimiento financiero reproducido extremo a extremo. Si en el futuro se publican metricas concretas, deben acompanarse de `seed`, rango temporal, conjunto de tickers, `config`, commit y comando exacto de reproduccion.

## Limitaciones conocidas

- El benchmark actual sigue siendo estadistico, no un backtest economico completo con costes, slippage y reglas operativas finales.
- La reproducibilidad total del entorno requiere lockfile y validacion en una matriz concreta de Python y CUDA.
- El codigo historico del repositorio todavia contiene modulos y comentarios heredados que conviene seguir limpiando.

## Aviso

Este proyecto no constituye asesoramiento financiero.
