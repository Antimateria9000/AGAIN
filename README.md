# Predictor bursatil con TFT

Este repositorio implementa un pipeline experimental de forecasting bursatil con Temporal Fusion Transformer (TFT), junto con una aplicacion Streamlit para inferencia, comparacion historica, benchmark estadistico y backtesting economico integrado sobre `again_econ`.

No es un sistema listo para inversion real. El alcance actual es investigacion aplicada, validacion tecnica del pipeline y reproducibilidad del flujo principal.

## Alcance actual

- Descarga de datos con yfinance
- Ingenieria de features tecnicas
- Entrenamiento TFT con cuantiles
- Inferencia desde la app Streamlit
- Comparacion historica sobre barras reales disponibles
- Benchmark estadistico con metricas MAPE, MAE, RMSE y DirAcc
- Backtesting economico desde la UI Streamlit usando `again_econ`
- Persistencia del historico de benchmark en SQLite
- Persistencia y catalogo de runs economicos en SQLite + parquet
- Smoke test reproducible con datos sinteticos

## Lo que NO hace hoy

- No debe interpretarse como un sistema de trading listo para produccion
- El backtesting economico integrado no equivale a un entrenamiento walk-forward estricto por ventana del modelo TFT
- No cubre live trading, intradia, short selling general ni ejecucion de mercado real

## Modulo economico `again_econ`

El repositorio incluye un modulo economico independiente llamado `again_econ`. Su alcance actual es una v1 seria, reproducible y desacoplada para evaluacion economica out-of-sample:

- walk-forward con ventanas de test economicamente independientes
- long-only sobre daily bars
- pipeline principal por provider: `ForecastProvider` o `SignalProvider`
- adaptador JSON de bundle solo como frontera de compatibilidad, no como centro del diseno
- adaptador oficial AGAIN TFT -> `again_econ` para ejecucion desde la app
- semantica temporal explicita por ventana: `train_start`, `train_end`, `test_start`, `test_end`, `lookahead_bars`, `execution_lag_bars` y `close_policy`
- timestamps operativos explicitados y validados: `observed_at`, `decision_timestamp`, `available_at`, `execution_timestamp`
- comisiones, slippage, scheduling, ranking y competencia de capital declarados y versionados
- manifests reproducibles por run y por ventana, con fingerprints y referencias a artefactos
- ledger, trades y snapshots reproducibles con invariantes contables verificables
- almacenamiento y reporting de corridas economicas para catalogo y lectura posterior desde Streamlit

Integracion actual en la app:

- seccion `Backtesting` en Streamlit
- modo `exploratory_live`: rapido, interactivo y util para inspeccion, pero no oficial
- modo `official_frozen`: persistente y reproducible como replay congelado del modelo activo, con snapshot de mercado y catalogo auditable

Importante:

- `official_frozen` no se presenta como WFO estricto del modelo TFT
- la app deja esto explicitado en UI, manifests y summary payloads

Garantias metodologicas de esta v1:

- no hay carry-over de posiciones entre ventanas
- `available_at` participa en el scheduling: una senal no puede ejecutarse antes de su disponibilidad real
- las senales cuya ejecucion cae fuera de la ventana se descartan de forma explicita y auditable
- las posiciones abiertas al final de la ventana se clausuran administrativamente en la ultima barra OOS
- la politica de competencia de capital ya no es implicita: queda declarada en config y persistida en manifests
- el resumen global OOS se calcula desde una curva chain-linked, no desde medias ingenuas de metricas por ventana

Limitaciones explicitas de `again_econ`:

- no sustituye al benchmark estadistico principal del pipeline TFT
- no soporta todavia intradia, margin, borrow, derivados ni politicas avanzadas de portfolio
- no es una mini-fork de PyBroker ni depende de internals fragiles del pipeline principal

Documentacion tecnica breve del modulo:

- `again_econ/README.md`

## Modulo de benchmark `again_benchmark`

El repositorio incluye un modulo nuevo e independiente llamado `again_benchmark` para benchmark estadistico reproducible y auditable. Su filosofia es separar por completo:

- definicion del benchmark
- snapshot de datos inmutable
- runner de corrida
- manifests de snapshot y run
- persistencia en artefactos + catalogo ligero
- adaptador fino de UI para Streamlit

`again_benchmark` soporta dos modos claramente distintos:

- `frozen`: benchmark oficial reproducible, ejecutado desde snapshot materializado y rerunnable desde artefactos congelados
- `live`: benchmark exploratorio, no oficial y no reproducible, pensado para inspeccion rapida

El benchmark legacy basado en `app/benchmark_utils.py` se conserva solo como compatibilidad exploratoria. No debe confundirse con el flujo oficial reproducible.

Garantias metodologicas actuales de `again_benchmark`:

- las definiciones oficiales se persisten como artefactos versionables en `benchmarks/definitions/*.yaml`
- cada snapshot oficial guarda parquet, checksum SHA-256 y `snapshot_manifest.json`
- cada run oficial guarda `run_manifest.json`, `summary.json`, `metrics.parquet`, `ticker_results.parquet` y `plot_payload.json`, todos con checksums
- el manifiesto de run persiste identidad de modelo, fingerprint de config, checksums de artefactos TFT disponibles, snapshot usado y estado de validacion
- el resumen global solo usa tickers efectivamente completados; cualquier descarte queda auditado con razon explicita
- el modo `live` queda marcado como exploratorio y no certificable; el modo `frozen` es el unico benchmark oficial reproducible

Limitaciones explicitas de `again_benchmark`:

- hoy implementa un corte reproducible comun por universo (`common_history_cutoff`); rolling-origin con anchors fijos queda preparado arquitectonicamente, pero no entra todavia en esta v1
- el rerun exacto reproducible aplica al modo `frozen`; el modo `live` puede reutilizar la misma interfaz, pero no la misma garantia metodologica

## Requisitos

- Python 3.11
- Entorno virtual recomendado
- Para GPU NVIDIA, instala la variante oficial CUDA de torch segun la documentacion de PyTorch

## Instalacion reproducible de referencia

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

El archivo `requirements.txt` fija un entorno de referencia para CPU. Si vas a usar CUDA, sustituye la instalacion de `torch` por la variante oficial compatible con tu hardware. Para este proyecto se ha dejado preparado:

- `requirements-torch-cu128.txt`
- `instalar_torch_cuda.bat`

El proyecto detecta automaticamente CUDA y hace fallback limpio a CPU si no esta disponible.

## Ejecucion

### Entrenamiento

```bash
python start_training.py --regions global --years 3
```

### Aplicacion Streamlit

```bash
streamlit run streamlit_app.py
```

La seccion `Entrenamiento` incorpora un bloque de mantenimiento del repositorio para:

- limpiar cache y temporales regenerables
- listar entrenamientos catalogados
- previsualizar rutas afectadas antes del borrado
- eliminar entrenamientos completos con confirmacion explicita si el perfil esta activo

## Tests

### Suite completa del repositorio

```bash
python -m pytest -q
```

### Smoke test del pipeline

Este test crea un dataset sintetico, entrena una ejecucion minima y valida inferencia y artefactos:

```bash
python -m pytest tests/integration/test_smoke_pipeline.py -q
```

## Configuracion

Archivo principal:

- `config/config.yaml`

Archivos de tickers:

- `config/tickers_with_names.yaml`
- `config/benchmark_tickers.yaml`

Artefactos y roots relevantes:

- `paths.artifacts_dir`
- `paths.training_artifacts_dir`
- `paths.backtest_storage_dir`
- `paths.cache_dir`
- `paths.tmp_dir`
- `paths.logs_root_dir`
- `data.raw_data_path`
- `data.processed_data_path`
- `data.train_processed_df_path`
- `data.val_processed_df_path`
- `paths.models_dir`
- `paths.normalizers_dir`
- `paths.logs_dir`
- `paths.training_catalog_path`
- `paths.benchmark_history_db_path`

## Estructura real del repositorio

```text
artifacts/
  backtests/
    econ/
  benchmarks/
  training/
    <profile_id>/
      active/
      runs/
app/
  app.py
  backtest_market_builder.py
  backtest_service.py
  benchmark_store.py
  benchmark_utils.py
  config_loader.py
  plot_utils.py
  services.py
again_benchmark/
  adapters/
  comparison.py
  config.py
  contracts.py
  definitions.py
  manifests.py
  metrics.py
  reports.py
  runner.py
  snapshots.py
  storage.py
  ui_adapter.py
  validation.py
config/
  backtests/
  benchmark_tickers.yaml
  config.yaml
  tickers_with_names.yaml
  runtime_profiles/
tests/
  app/
  again_benchmark/
  again_econ/
  helpers/
  integration/
  scripts/
var/
  cache/
  logs/
  tmp/
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
again_econ/
start_training.py
  requirements.txt
  requirements-dev.txt
  pyproject.toml
  README.md
```

## Politica de artefactos

Los siguientes ficheros y carpetas son artefactos derivados y no fuente de verdad del repositorio:

- `artifacts/`
- `var/`
- `data/`
- `logs/`
- `lightning_logs/`
- checkpoints legacy o locales en `models/`

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
- La GPU solo se usara si la instalacion de torch dentro de `.venv` incluye soporte CUDA y `torch.cuda.is_available()` devuelve `True`

## Aviso

Este proyecto no constituye asesoramiento financiero.
