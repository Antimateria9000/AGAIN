# again_econ

`again_econ` es el modulo economico independiente del repositorio AGAIN. Su objetivo es evaluar comportamiento economico out-of-sample sin acoplarse a internals fragiles del pipeline TFT y sin intentar replicar el alcance de una plataforma completa de backtesting.

## Arquitectura

Pipeline principal:

1. materializacion walk-forward de ventanas
2. provider por ventana (`ForecastProvider` o `SignalProvider`)
3. normalizacion del payload
4. traduccion de forecasts a senales, si aplica
5. scheduling operativo
6. ejecucion y contabilidad
7. agregacion de resultados y manifests

El bundle JSON sigue existiendo, pero solo como adaptador estable de compatibilidad (`AgainBundleAdapter`).

## Semantica temporal

Cada `WalkforwardWindow` declara explicitamente:

- `train_start`
- `train_end`
- `test_start`
- `test_end`
- `lookahead_bars`
- `execution_lag_bars`
- `close_policy`

Cada forecast o signal puede declarar:

- `observed_at`: momento de observacion economica del dato base
- `decision_timestamp`: momento de decision economica declarado por la estrategia
- `available_at`: momento real en que el payload estuvo disponible para el motor
- `execution_timestamp`: momento en que la orden se ejecuta realmente

Reglas metodologicas relevantes:

- `observed_at <= decision_timestamp`
- `observed_at <= available_at`
- una senal nunca se ejecuta antes de `max(decision_timestamp, available_at)`
- la ejecucion se desplaza por `execution_lag_bars`
- si la ejecucion cae fuera de la ventana, la senal se descarta con razon explicita

## Providers y adaptadores

La ruta principal del runner es `run_backtest_with_provider(...)`.

Opciones:

- `ForecastProvider`: entrega forecasts por ventana
- `SignalProvider`: entrega signals por ventana
- `run_backtest(...)`: adaptador fino para tuplas directas
- `run_backtest_from_bundle(...)`: adaptador fino para bundles JSON

## Policies

Las policies relevantes quedan declaradas y versionadas:

- traduccion de forecasts a signals
- scheduling
- sizing
- capital competition
- artifact policy

La politica de competencia de capital ya no es una consecuencia oculta del orden lexicografico. Puede declararse, por ejemplo, como `instrument_asc` o `score_desc`.

## Reproducibilidad

Cada corrida produce un `RunManifest` con:

- identidad y version del provider
- versions de signal/execution/artifact policies
- fingerprints de config, mercado, plan de ventanas e inputs
- manifests por ventana
- referencias a artefactos de entrada
- conteo y razones de descarte

Cada `WindowManifest` resume:

- rango train/test
- lookahead y execution lag
- payload fingerprint
- provider identity
- conteos de records, senales, fills, trades y descartes

## Testing

La suite de `tests/test_again_econ_*` cubre:

- invariantes temporales
- scheduling con `available_at`
- providers mock por ventana
- ranking y competencia de capital
- invariantes contables del ledger
- manifests y reproducibilidad
- regresion de la curva OOS chain-linked

## Alcance no incluido

Queda fuera de esta v1:

- live trading
- intradia complejo
- margin, borrow y derivados
- emular plataformas generalistas de backtesting
