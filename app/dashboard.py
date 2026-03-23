"""
app/dashboard.py

Дашборд мониторинга RAG-бота "Хронорус".

Запуск:
    streamlit run app/dashboard.py

Требования (добавить в requirements.txt):
    streamlit>=1.35
    pandas>=2.0
    plotly>=5.20
    watchdog>=4.0   # для авто-перезагрузки при обновлении CSV
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Константы ─────────────────────────────────────────────────────────────────
CSV_PATH = Path("logs/rag_metrics.csv")
REFRESH_INTERVAL_SEC = 30          # авто-обновление (0 = выключено)
PAGE_SIZE = 50                     # строк в таблице на одной странице

# Цвета меток (соответствуют ResponseLabel)
LABEL_COLORS = {
    "success":       "#4CAF93",
    "off_topic":     "#F4A261",
    "after_2014":    "#E76F51",
    "no_info_in_db": "#A8DADC",
    "error":         "#E63946",
}
LABEL_NAMES = {
    "success":       "✅ Ответил",
    "off_topic":     "🚫 Не по теме",
    "after_2014":    "📅 После 2014",
    "no_info_in_db": "🔍 Нет в базе",
    "error":         "⚠️ Ошибка",
}

# ── Страница ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Хронорус · Мониторинг",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Кастомные стили ───────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        letter-spacing: -0.02em;
    }

    /* Шапка */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '⚔';
        position: absolute;
        right: 2rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 5rem;
        opacity: 0.07;
    }
    .hero h1 {
        color: #e8d5b7 !important;
        font-size: 2rem;
        margin: 0 0 .25rem 0;
    }
    .hero p {
        color: rgba(232, 213, 183, 0.6);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        margin: 0;
    }

    /* KPI-карточки */
    .kpi-card {
        background: #1e1e2e;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
    }
    .kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #e8d5b7;
        line-height: 1;
        margin-bottom: .3rem;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: rgba(232,213,183,0.5);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.07) !important; }

    /* Таблица */
    .stDataFrame { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }

    /* Сайдбар */
    section[data-testid="stSidebar"] {
        background: #12121f !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Загрузка данных ───────────────────────────────────────────────────────────
@st.cache_data(ttl=REFRESH_INTERVAL_SEC if REFRESH_INTERVAL_SEC else None)
def load_data(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Распаковываем список скоров из строки
    def parse_scores(val):
        try:
            return ast.literal_eval(val) if isinstance(val, str) else []
        except Exception:
            return []

    df["scores_list"] = df["retrieved_docs_scores"].apply(parse_scores)
    df["label_name"] = df["label"].map(LABEL_NAMES).fillna(df["label"])
    return df


# ── Сайдбар ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Фильтры")

    csv_path_input = st.text_input("Путь к CSV", value=str(CSV_PATH))

    df_raw = load_data(csv_path_input)
    if df_raw.empty:
        st.warning("CSV пуст или не найден.")
    else:
        # Диапазон дат
        min_date = df_raw["timestamp"].min().date()
        max_date = df_raw["timestamp"].max().date()
        date_range = st.date_input(
            "Период",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        # Метки
        all_labels = sorted(df_raw["label"].dropna().unique().tolist())
        selected_labels = st.multiselect(
            "Метки",
            options=all_labels,
            default=all_labels,
            format_func=lambda x: LABEL_NAMES.get(x, x),
        )

        # Порог времени ответа
        max_rt = float(df_raw["response_time_sec"].max()) if "response_time_sec" in df_raw else 60.0
        rt_threshold = st.slider(
            "Макс. время ответа (с)",
            0.0, max(max_rt, 1.0), max(max_rt, 1.0), 0.1,
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-family:IBM Plex Mono;font-size:0.7rem;color:rgba(255,255,255,0.3)'>"
        "Хронорус · Monitoring v1.0</div>",
        unsafe_allow_html=True,
    )

    if REFRESH_INTERVAL_SEC:
        st.markdown(
            f"<div style='font-family:IBM Plex Mono;font-size:0.7rem;color:rgba(255,255,255,0.3)'>"
            f"Авто-обновление: {REFRESH_INTERVAL_SEC}с</div>",
            unsafe_allow_html=True,
        )

# ── Шапка ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>📜 Хронорус · Мониторинг</h1>
        <p>RAG-метрики · История России до 2014 года</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Проверяем данные ──────────────────────────────────────────────────────────
if df_raw.empty:
    st.info("Нет данных для отображения. Запустите бота и задайте несколько вопросов.")
    st.stop()

# ── Применяем фильтры ─────────────────────────────────────────────────────────
df = df_raw.copy()

if len(date_range) == 2:
    start_dt = pd.Timestamp(date_range[0], tz="UTC")
    end_dt   = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)]

if selected_labels:
    df = df[df["label"].isin(selected_labels)]

df = df[df["response_time_sec"] <= rt_threshold]

if df.empty:
    st.warning("По заданным фильтрам данных нет.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# KPI-карточки
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### Ключевые показатели")

k1, k2, k3, k4, k5 = st.columns(5)

success_rate = (df["label"] == "success").mean() * 100
avg_rt       = df["response_time_sec"].mean()
p95_rt       = df["response_time_sec"].quantile(0.95)
avg_docs     = df["retrieved_docs_count"].mean()
total_req    = len(df)

for col, val, label in [
    (k1, f"{total_req:,}", "Запросов"),
    (k2, f"{success_rate:.1f}%", "Успешных ответов"),
    (k3, f"{avg_rt:.2f}с", "Среднее время"),
    (k4, f"{p95_rt:.2f}с", "P95 время"),
    (k5, f"{avg_docs:.1f}", "Документов (avg)"),
]:
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{val}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Строка 1: Распределение меток + Время ответа
# ═══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1.6])

with col_left:
    st.markdown("#### Распределение меток")
    label_counts = (
        df["label"]
        .value_counts()
        .reset_index()
        .rename(columns={"label": "label", "count": "count"})
    )
    label_counts["name"] = label_counts["label"].map(LABEL_NAMES).fillna(label_counts["label"])
    label_counts["color"] = label_counts["label"].map(LABEL_COLORS).fillna("#888")

    fig_pie = go.Figure(
        go.Pie(
            labels=label_counts["name"],
            values=label_counts["count"],
            marker_colors=label_counts["color"],
            hole=0.55,
            textinfo="percent",
            textfont_size=13,
            hovertemplate="<b>%{label}</b><br>%{value} запросов (%{percent})<extra></extra>",
        )
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            font=dict(color="#aaa", size=11),
            bgcolor="rgba(0,0,0,0)",
            orientation="v",
        ),
        margin=dict(t=10, b=10, l=10, r=10),
        height=280,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.markdown("#### Время ответа по меткам")
    fig_box = go.Figure()
    for lbl in df["label"].unique():
        sub = df[df["label"] == lbl]["response_time_sec"]
        fig_box.add_trace(
            go.Box(
                y=sub,
                name=LABEL_NAMES.get(lbl, lbl),
                marker_color=LABEL_COLORS.get(lbl, "#888"),
                boxmean="sd",
                line_width=1.5,
            )
        )
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,35,0.6)",
        font=dict(color="#aaa", size=11),
        yaxis=dict(
            title="Время (с)",
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=280,
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Строка 2: Временной ряд запросов + Скоры документов
# ═══════════════════════════════════════════════════════════════════════════════
col_a, col_b = st.columns([1.6, 1])

with col_a:
    st.markdown("#### Запросы во времени")

    df_ts = df.set_index("timestamp").sort_index()
    freq_options = {"1 мин": "1min", "5 мин": "5min", "15 мин": "15min", "1 час": "1h", "1 день": "1D"}
    chosen_freq_label = st.select_slider(
        "Агрегация", options=list(freq_options.keys()), value="15 мин", label_visibility="collapsed"
    )
    freq = freq_options[chosen_freq_label]

    agg = (
        df_ts.groupby([pd.Grouper(freq=freq), "label"])
        .size()
        .reset_index(name="count")
    )

    fig_ts = px.bar(
        agg,
        x="timestamp",
        y="count",
        color="label",
        color_discrete_map=LABEL_COLORS,
        labels={"timestamp": "", "count": "Запросов", "label": "Метка"},
        custom_data=["label"],
    )
    fig_ts.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>%{y} запросов<br>%{x}<extra></extra>"
    )
    fig_ts.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,35,0.6)",
        font=dict(color="#aaa", size=11),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Запросов"),
        legend=dict(
            title="",
            font=dict(color="#aaa", size=10),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            y=-0.2,
        ),
        margin=dict(t=10, b=10, l=10, r=10),
        height=280,
        bargap=0.15,
    )
    # Переименовываем легенду
    for trace in fig_ts.data:
        trace.name = LABEL_NAMES.get(trace.name, trace.name)
    st.plotly_chart(fig_ts, use_container_width=True)

with col_b:
    st.markdown("#### Скоры retrieved документов")

    all_scores = [s for scores in df["scores_list"] for s in scores]
    if all_scores:
        fig_hist = go.Figure(
            go.Histogram(
                x=all_scores,
                nbinsx=30,
                marker_color="#4CAF93",
                marker_line_color="rgba(0,0,0,0.3)",
                marker_line_width=0.5,
                opacity=0.85,
                hovertemplate="Скор: %{x:.3f}<br>Кол-во: %{y}<extra></extra>",
            )
        )
        # Линия медианы
        median_score = pd.Series(all_scores).median()
        fig_hist.add_vline(
            x=median_score,
            line_dash="dash",
            line_color="#E8D5B7",
            annotation_text=f"медиана {median_score:.3f}",
            annotation_font_color="#E8D5B7",
            annotation_font_size=10,
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,35,0.6)",
            font=dict(color="#aaa", size=11),
            xaxis=dict(title="Score", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Частота", gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Нет данных о скорах.")


# ═══════════════════════════════════════════════════════════════════════════════
# Строка 3: Количество документов + Скор vs Время ответа
# ═══════════════════════════════════════════════════════════════════════════════
col_c, col_d = st.columns(2)

with col_c:
    st.markdown("#### Количество retrieved документов")
    doc_counts = df["retrieved_docs_count"].value_counts().sort_index().reset_index()
    doc_counts.columns = ["docs", "count"]
    fig_docs = px.bar(
        doc_counts,
        x="docs",
        y="count",
        color_discrete_sequence=["#A8DADC"],
        labels={"docs": "Документов", "count": "Запросов"},
    )
    fig_docs.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,35,0.6)",
        font=dict(color="#aaa", size=11),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", dtick=1),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
    )
    st.plotly_chart(fig_docs, use_container_width=True)

with col_d:
    st.markdown("#### Макс. скор документа vs Время ответа")
    df_scatter = df.dropna(subset=["max_doc_score"]).copy()
    if not df_scatter.empty:
        fig_scatter = px.scatter(
            df_scatter,
            x="max_doc_score",
            y="response_time_sec",
            color="label",
            color_discrete_map=LABEL_COLORS,
            opacity=0.7,
            hover_data={"user_query": True, "label": True},
            labels={
                "max_doc_score": "Макс. скор",
                "response_time_sec": "Время (с)",
                "label": "Метка",
            },
        )
        for trace in fig_scatter.data:
            trace.name = LABEL_NAMES.get(trace.name, trace.name)
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,35,0.6)",
            font=dict(color="#aaa", size=11),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(
                title="",
                font=dict(color="#aaa", size=10),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(t=10, b=10, l=10, r=10),
            height=260,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Нет данных о скорах для скаттера.")


# ═══════════════════════════════════════════════════════════════════════════════
# Строка 4: Тренд скользящего среднего времени ответа
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("#### Тренд времени ответа (скользящее среднее)")
window = st.slider("Окно (запросов)", 5, 100, 20, key="ma_window", label_visibility="collapsed")

df_sorted = df.sort_values("timestamp").reset_index(drop=True)
df_sorted["ma_rt"] = df_sorted["response_time_sec"].rolling(window=window, min_periods=1).mean()

fig_ma = go.Figure()
fig_ma.add_trace(
    go.Scatter(
        x=df_sorted["timestamp"],
        y=df_sorted["response_time_sec"],
        mode="markers",
        marker=dict(size=4, color="rgba(168,218,220,0.4)"),
        name="Факт",
    )
)
fig_ma.add_trace(
    go.Scatter(
        x=df_sorted["timestamp"],
        y=df_sorted["ma_rt"],
        mode="lines",
        line=dict(color="#E8D5B7", width=2.5),
        name=f"MA({window})",
    )
)
fig_ma.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(20,20,35,0.6)",
    font=dict(color="#aaa", size=11),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    yaxis=dict(title="Время (с)", gridcolor="rgba(255,255,255,0.06)"),
    legend=dict(font=dict(color="#aaa"), bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=10, b=10, l=10, r=10),
    height=240,
)
st.plotly_chart(fig_ma, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Таблица с последними запросами (с поиском)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🔎 Лог запросов")

search_col, label_col = st.columns([3, 1])
with search_col:
    search_query = st.text_input("Поиск по тексту вопроса / ответа", placeholder="введи ключевое слово…")
with label_col:
    filter_label = st.selectbox(
        "Фильтр по метке",
        options=["Все"] + list(LABEL_NAMES.keys()),
        format_func=lambda x: "Все" if x == "Все" else LABEL_NAMES[x],
    )

df_table = df.sort_values("timestamp", ascending=False).copy()

if search_query:
    mask = (
        df_table["user_query"].str.contains(search_query, case=False, na=False)
        | df_table["bot_response"].str.contains(search_query, case=False, na=False)
    )
    df_table = df_table[mask]

if filter_label != "Все":
    df_table = df_table[df_table["label"] == filter_label]

# Выбираем и переименовываем столбцы
display_cols = {
    "timestamp":           "Время",
    "user_id":             "User ID",
    "user_query":          "Вопрос",
    "bot_response":        "Ответ",
    "response_time_sec":   "Время (с)",
    "retrieved_docs_count":"Документов",
    "avg_doc_score":       "Avg score",
    "max_doc_score":       "Max score",
    "label_name":          "Метка",
    "error_message":       "Ошибка",
}

df_display = df_table[list(display_cols.keys())].rename(columns=display_cols)
df_display["Время (с)"] = df_display["Время (с)"].round(3)
df_display["Avg score"] = df_display["Avg score"].round(3)
df_display["Max score"] = df_display["Max score"].round(3)
df_display["Время"] = df_display["Время"].dt.strftime("%Y-%m-%d %H:%M:%S")

st.caption(f"Показано {len(df_display):,} из {len(df):,} записей")
st.dataframe(df_display.head(PAGE_SIZE), use_container_width=True, height=420)

if len(df_display) > PAGE_SIZE:
    st.caption(f"Показаны первые {PAGE_SIZE} записей. Используй фильтры для уточнения.")


# ═══════════════════════════════════════════════════════════════════════════════
# Экспорт
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
dl_col, _, _ = st.columns([1, 2, 2])
with dl_col:
    csv_bytes = df_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Скачать отфильтрованный CSV",
        data=csv_bytes,
        file_name="chronorus_metrics_export.csv",
        mime="text/csv",
    )

# ── Авто-обновление ───────────────────────────────────────────────────────────
if REFRESH_INTERVAL_SEC:
    time.sleep(REFRESH_INTERVAL_SEC)
    st.cache_data.clear()
    st.rerun()
