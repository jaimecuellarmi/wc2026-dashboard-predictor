# Predictor de partidos — Mundial 2026

Estima el resultado de un partido del Mundial 2026: probabilidades 1X2, over/under,
ambos marcan, marcadores más probables y comparación con cuotas.

## Cómo funciona
- **Motor**: regresión de Poisson ajustada por rival (modelo tipo Dixon-Coles) sobre
  todos los partidos internacionales desde 2022, con más peso a lo reciente. Estima
  ataque y defensa de cada selección controlando por la calidad del rival.
- **Prior**: las valoraciones de goles se mezclan con un prior de **ranking FIFA +
  valor de plantel**, para anclar a equipos con datos ruidosos (rachas, rivales débiles).
  El peso del prior es un **slider** (`theta`): 0 = solo datos de goles, 1 = solo
  ranking/valor.
- Con los goles esperados de cada lado se construye la distribución conjunta de
  marcadores (Poisson) y de ahí salen todos los mercados.

## Archivos
- `app.py` — la aplicación Streamlit.
- `wc2026_model.json` — ataque/defensa (datos + prior) por selección y parámetros globales.
- `wc2026_countries.csv` — contexto (ranking, valor, forma, histórico).
- `wc2026_fixtures.csv` — calendario de fase de grupos.
- `wc2026_players.csv` — opcional (contexto de jugadores).
- `build_model.py` — entrena el modelo y regenera el JSON + fixtures (correr local).

## Correr local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Regenerar el modelo
```bash
python build_model.py    # necesita scikit-learn (Anaconda lo trae) y wc2026_countries.csv
```
Esto baja los resultados internacionales actualizados, reentrena y reescribe
`wc2026_model.json` y `wc2026_fixtures.csv`. Útil para actualizar a medida que avanza
el Mundial. Luego se vuelve a desplegar (re-subir a GitHub).

## Aviso
Es una estimación estadística de apoyo, no una garantía. Juega con responsabilidad.
