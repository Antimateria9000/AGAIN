\# AGENTS.md



\## Prioridades absolutas



1\. Evitar cualquier tipo de lookahead bias, leakage temporal o contaminación entre train/validación/inferencia.

2\. Preservar reproducibilidad funcional y compatibilidad de artefactos.

3\. Mantener coherencia entre:

&#x20;  - config/config.yaml

&#x20;  - scripts/utils/data\_schema.py

&#x20;  - scripts/preprocessor.py

&#x20;  - scripts/train.py

&#x20;  - scripts/prediction\_engine.py

&#x20;  - scripts/utils/model\_readiness.py

4\. No presentar benchmarks estadísticos como si fueran backtests económicos finales.

5\. Hacer cambios mínimos, localizados y justificables.



\## Reglas metodológicas obligatorias



\### Sobre forecasting financiero

\- Distingue siempre entre:

&#x20; - forecasting estadístico,

&#x20; - comparación histórica,

&#x20; - benchmark de error/dirección,

&#x20; - backtest económico real,

&#x20; - decisión de inversión.

\- Si propones mejoras metodológicas, prioriza robustez, validez temporal y reproducibilidad por encima de complejidad cosmética.



\### Sobre leakage y validez temporal

\- Nunca uses información futura para construir features del presente.

\- Nunca mezcles barras posteriores al corte de validación en el conjunto de entrenamiento.

\- Nunca cambies el split temporal sin revisar su impacto en:

&#x20; - scripts/preprocessor.py

&#x20; - tests

&#x20; - benchmark histórico

&#x20; - interpretación de métricas

\- Si tocas ingeniería de features, comprueba explícitamente que todas las transformaciones usan solo historia pasada o presente.

\- Si introduces nuevas features, documenta su ventana, lag y riesgo de leakage.



\### Sobre benchmark e inferencia

\- Si modificas benchmark o comparación histórica, conserva trazabilidad y separa claramente:

&#x20; - precios reales,

&#x20; - precios predichos,

&#x20; - retornos paso a paso,

&#x20; - métricas agregadas.

\- Si una métrica puede inducir a error en contexto financiero, indícalo.



\## Reglas de artefactos y compatibilidad



\- Trata los artefactos persistidos como sensibles al esquema:

&#x20; - models/\*.pth

&#x20; - models/normalizers/\*.pkl

&#x20; - data/train/processed\_dataset.pt

&#x20; - checksums .sha256

&#x20; - metadata .meta.json

\- No rompas compatibilidad de artefactos sin dejarlo explícito.

\- Si cambias cualquier elemento que afecte al schema hash, embeddings, features, categoricals o target:

&#x20; - asume que puede requerir regenerar normalizadores,

&#x20; - regenerar dataset procesado,

&#x20; - reentrenar checkpoint,

&#x20; - actualizar tests.

\- No des por válidos checkpoints legacy si faltan checksums o metadatos compatibles.

\- No desactives validación de checksums o metadata salvo que el objetivo del cambio sea precisamente migración controlada de artefactos, y documentándolo.



\## Reglas de configuración



\- La fuente de verdad principal es config/config.yaml.

\- Si tocas hiperparámetros, revisa su impacto en:

&#x20; - scripts/model.py

&#x20; - scripts/utils/data\_schema.py

&#x20; - scripts/prediction\_engine.py

&#x20; - tests

\- No cambies:

&#x20; - max\_encoder\_length

&#x20; - min\_encoder\_length

&#x20; - max\_prediction\_length

&#x20; - embedding\_sizes

&#x20; - sectors

&#x20; - target / numeric features / categoricals

&#x20; sin revisar compatibilidad completa del pipeline.

\- Si añades nuevas claves de config:

&#x20; - valídalas en el esquema correspondiente,

&#x20; - dales defaults seguros,

&#x20; - evita romper configs existentes.



\## Reglas de código



\- Haz cambios pequeños y bien acotados.

\- No refactorices masivamente si el usuario no lo pide.

\- No cambies nombres públicos, rutas de artefactos o contratos de funciones sin necesidad real.

\- No introduzcas dependencias nuevas salvo motivo fuerte.

\- Prefiere claridad, trazabilidad y seguridad frente a “magia”.

\- Mantén compatibilidad con Python 3.11 salvo instrucción expresa en contra.



\## Qué revisar antes de editar



Antes de modificar cualquier parte importante, identifica primero:

\- flujo de datos desde descarga hasta inferencia,

\- target exacto del modelo,

\- features activas,

\- split temporal,

\- artefactos que dependen del esquema,

\- tests que cubren esa parte.



\## Qué revisar antes de editar



Cuando hagas cambios, informa siempre de:

* archivos modificados,
* motivo técnico de cada cambio,
* riesgos metodológicos,
* si exige reentrenar o regenerar artefactos,
* qué tests o checks deben ejecutarse,
* si afecta a validez de benchmark o inferencia.



