# ============================================================
# app.py — Predictor de partidos Mundial 2026
# Poisson ajustado por rival + prior de ranking/valor (slider).
# Reads: wc2026_model.json, wc2026_countries.csv, wc2026_fixtures.csv,
#        wc2026_players.csv (opcional). Run: streamlit run app.py
# ============================================================
import json
import math
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Predictor Mundial 2026", page_icon="🎯", layout="wide")

MAXG = 10  # goals grid for the Poisson matrix

# ----------------------------- Data -----------------------------
@st.cache_data
def load():
    with open("wc2026_model.json", encoding="utf-8") as f:
        model = json.load(f)
    countries = pd.read_csv("wc2026_countries.csv")
    fixtures = pd.read_csv("wc2026_fixtures.csv") if os.path.exists("wc2026_fixtures.csv") else None
    players = pd.read_csv("wc2026_players.csv") if os.path.exists("wc2026_players.csv") else None
    return model, countries, fixtures, players

for needed in ["wc2026_model.json", "wc2026_countries.csv"]:
    if not os.path.exists(needed):
        st.error(f"Falta {needed}. Corre la app desde la carpeta con los archivos del modelo.")
        st.stop()

model, countries, fixtures, players = load()
TEAMS = sorted(model["teams"].keys())
cinfo = countries.set_index("Country")

# ----------------------------- Model math -----------------------------
_FACT = [math.factorial(k) for k in range(MAXG + 1)]

def pois_pmf(lam):
    ks = np.arange(MAXG + 1)
    return np.exp(-lam) * lam ** ks / np.array(_FACT)

def ratings(team, theta):
    t = model["teams"][team]
    atk = (1 - theta) * t["a_data"] + theta * t["a_prior"]
    dfn = (1 - theta) * t["d_data"] + theta * t["d_prior"]
    return atk, dfn

def predict(home, away, theta, neutral=True):
    intc, hc = model["meta"]["intercept"], model["meta"]["home_coef"]
    aH, dH = ratings(home, theta)
    aA, dA = ratings(away, theta)
    xh = np.exp(intc + aH + dA + (0 if neutral else hc))
    xa = np.exp(intc + aA + dH)
    M = np.outer(pois_pmf(xh), pois_pmf(xa))   # joint scoreline prob (independence)
    tot = np.add.outer(np.arange(MAXG + 1), np.arange(MAXG + 1))
    return {
        "xh": xh, "xa": xa, "M": M,
        "pH": float(np.tril(M, -1).sum()), "pD": float(np.trace(M)), "pA": float(np.triu(M, 1).sum()),
        "o15": float(M[tot > 1.5].sum()), "o25": float(M[tot > 2.5].sum()), "o35": float(M[tot > 3.5].sum()),
        "btts": float(M[1:, 1:].sum()),
    }

def pct(x):
    return f"{x*100:.0f}%"

# ----------------------------- Sidebar -----------------------------
st.sidebar.title("🎯 Predictor")
theta = st.sidebar.slider(
    "Peso del prior (ranking + valor)", 0.0, 1.0, 0.5, 0.05,
    help="0 = solo datos de goles (forma/resultados). 1 = solo ranking + valor de plantel. "
         "El resto mezcla ambos.")
neutral = st.sidebar.checkbox("Sede neutral", value=True,
    help="Desmárcalo solo si el primer equipo juega de local (selecciones anfitrionas).")
st.sidebar.markdown("---")
st.sidebar.caption(f"Modelo entrenado: {model['meta']['fitted_on']} · "
                   f"datos desde {model['meta']['window_start']} · "
                   f"ventaja local ×{np.exp(model['meta']['home_coef']):.2f}")

# ----------------------------- Match selection -----------------------------
st.header("Predictor de partidos — Mundial 2026")

c1, c2 = st.columns(2)
home = c1.selectbox("Equipo 1", TEAMS, index=TEAMS.index("Spain") if "Spain" in TEAMS else 0)
away = c2.selectbox("Equipo 2", TEAMS, index=TEAMS.index("Brazil") if "Brazil" in TEAMS else 1)

if home == away:
    st.warning("Elige dos selecciones distintas.")
    st.stop()

r = predict(home, away, theta, neutral)

# ----------------------------- 1X2 + xG -----------------------------
st.subheader(f"{home}  vs  {away}")
c1, c2, c3 = st.columns(3)
c1.metric(f"Gana {home}", pct(r["pH"]))
c2.metric("Empate", pct(r["pD"]))
c3.metric(f"Gana {away}", pct(r["pA"]))
st.caption(f"Goles esperados (xG):  {home} {r['xh']:.2f}  ·  {away} {r['xa']:.2f}")

# ----------------------------- Markets -----------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Over 2.5", pct(r["o25"]));  m1.caption(f"Under 2.5: {pct(1-r['o25'])}")
m2.metric("Over 1.5", pct(r["o15"]))
m3.metric("Over 3.5", pct(r["o35"]))
m4.metric("Ambos marcan", pct(r["btts"]))

# ----------------------------- Scorelines -----------------------------
left, right = st.columns([3, 2])
with left:
    st.markdown("**Mapa de marcadores** (prob. de cada resultado)")
    g = 6
    sub = r["M"][:g+1, :g+1]
    df = pd.DataFrame([
        {"home_goals": i, "away_goals": j, "p": sub[i, j]*100}
        for i in range(g+1) for j in range(g+1)])
    heat = alt.Chart(df).mark_rect().encode(
        x=alt.X("away_goals:O", title=f"Goles {away}"),
        y=alt.Y("home_goals:O", title=f"Goles {home}", sort="descending"),
        color=alt.Color("p:Q", scale=alt.Scale(scheme="greens"), legend=alt.Legend(title="%")),
        tooltip=[alt.Tooltip("home_goals:O", title=home),
                 alt.Tooltip("away_goals:O", title=away),
                 alt.Tooltip("p:Q", title="prob %", format=".1f")],
    ).properties(height=320)
    st.altair_chart(heat, use_container_width=True)
with right:
    st.markdown("**Marcadores más probables**")
    flat = sorted(((r["M"][i, j], i, j) for i in range(7) for j in range(7)), reverse=True)[:8]
    sc = pd.DataFrame([{"Marcador": f"{i}–{j}", "Prob.": pct(p)} for p, i, j in flat])
    st.dataframe(sc, hide_index=True, use_container_width=True)

# ----------------------------- Team context -----------------------------
with st.expander("Contexto de las selecciones"):
    def info_block(t):
        if t not in cinfo.index:
            return
        x = cinfo.loc[t]
        st.markdown(f"**{t}**")
        st.write(f"Ranking FIFA: {int(x['fifa_rank'])}")
        if "squad_value_m" in x: st.write(f"Valor de plantel: €{x['squad_value_m']:.0f}m")
        if "last10_form" in x:   st.write(f"Forma (últimos 10): {x['last10_form']}  ({x['last10_ppg']} ppg)")
        if "wc_best_finish" in x: st.write(f"Mejor en Mundial: {x['wc_best_finish']}")
    a, b = st.columns(2)
    with a: info_block(home)
    with b: info_block(away)

# ----------------------------- Value vs odds -----------------------------
with st.expander("Comparar con cuotas (valor)"):
    st.caption("Mete la cuota decimal de la casa. Si la probabilidad del modelo supera a la "
               "implícita por la cuota, hay valor (positivo).")
    cols = st.columns(4)
    labels = [(f"1 ({home})", r["pH"]), ("X", r["pD"]), (f"2 ({away})", r["pA"]), ("Over 2.5", r["o25"])]
    for col, (lab, p) in zip(cols, labels):
        with col:
            odd = st.number_input(lab, min_value=1.01, value=2.00, step=0.05, key="odd_"+lab)
            implied = 1 / odd
            edge = p - implied
            st.metric("Modelo / implícita", f"{pct(p)} / {pct(implied)}",
                      delta=f"{edge*100:+.0f} pp", delta_color="normal")

st.markdown("---")
st.caption("Modelo Poisson ajustado por rival, mezclado con un prior de ranking + valor de plantel. "
           "Basado en internacionales desde 2022 con peso a lo reciente. Es una estimación estadística, "
           "no una garantía — úsalo como apoyo, no como certeza. Juega con responsabilidad.")
