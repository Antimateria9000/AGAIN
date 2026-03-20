# Predictor bursatil con TFT

Este repositorio implementa un pipeline experimental de forecasting bursatil con Temporal Fusion Transformer (TFT), junto con una aplicacion Streamlit para inferencia, comparacion historica y benchmark estadistico.

No es un sistema listo para inversion real. El alcance actual es investigacion aplicada, validacion tecnica del pipeline y reproducibilidad del flujo principal.

## Alcance actual

- Descarga de datos con yfinance
- Ingenieria de features tecnicas
- Entrenamiento TFT con cuantiles
- Inferencia desde la app Streamlit
- Comparacion historica sobre barras reales disponibles
- Benchmark estadistico con metricas MAPE, MAE, RMSE y DirAcc
- Persistencia del historico de benchmark en SQLite
- Smoke test reproducible con datos sinteticos

## Lo que NO hace hoy

- No implementa un backtest economico completo
- No incluye costes, slippage ni reglas operativas finales
- No debe interpretarse como un sistema de trading listo para produccion

## Requisitos

- Python 3.11
- Entorno virtual recomendado
- Para GPU, instala la variante oficial de torch segun la documentacion de PyTorch

## Instalacion reproducible de referencia

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

El archivo `requirements.txt` fija un entorno de referencia para CPU. Si vas a usar CUDA, debes sustituir la instalacion de `torch` por la variante oficial compatible con tu hardware.

## Ejecucion

### Entrenamiento

```bash
python start_training.py --regions global --years 3
```

### Aplicacion Streamlit

```bash
streamlit run app/app.py
```

## Tests

### Suite completa del repositorio

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Smoke test del pipeline

Este test crea un dataset sintetico, entrena una ejecucion minima y valida inferencia y artefactos:

```bash
python -m unittest tests.test_smoke_pipeline -v
```

## Configuracion

Archivo principal:

- `config/config.yaml`

Archivos de tickers:

- `config/tickers_with_names.yaml`
- `config/benchmark_tickers.yaml`

Artefactos relevantes:

- `data/raw_data_path`
- `data/processed_data_path`
- `data/train_processed_df_path`
- `data/val_processed_df_path`
- `paths/models_dir`
- `paths/normalizers_dir`
- `paths/logs_dir`
- `paths/benchmark_history_db_path`

## Estructura real del repositorio

```text
app/
  app.py
  benchmark_store.py
  benchmark_utils.py
  config_loader.py
  plot_utils.py
  services.py
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
start_training.py
tests/
requirements.txt
requirements-dev.txt
pyproject.toml
README.md
```

## Politica de artefactos

Los siguientes ficheros y carpetas son artefactos derivados y no fuente de verdad del repositorio:

- `data/`
- `logs/`
- `lightning_logs/`
- checkpoints locales en `models/`

Los artefactos persistidos se validan con checksum SHA-256. Si el checksum falta o no coincide, la carga debe considerarse no fiable.

## Reproducibilidad

La reproducibilidad funcional minima exigible del proyecto pasa por:

- entorno Python 3.11 consistente
- dependencias fijadas
- smoke test verde
- CI verde
- mismos datos, seed, config y commit

Si se publican metricas concretas, deben ir acompanadas de seed, rango temporal, conjunto de tickers, config, commit y comando exacto de reproduccion.

## Limitaciones conocidas

- El benchmark actual sigue siendo estadistico, no un backtest economico final
- Las fechas futuras de la vista de prediccion usan dias habiles aproximados, no calendario bursatil por mercado
- Los artefactos legacy del repositorio pueden requerir regeneracion si no tienen checksums o metadatos compatibles

## Aviso

Este proyecto no constituye asesoramiento financiero.
