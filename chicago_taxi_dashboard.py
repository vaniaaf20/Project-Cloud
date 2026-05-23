"""
Dashboard Streamlit — Chicago Taxi Trips 2023
Jalankan: streamlit run chicago_taxi_dashboard.py
Letakkan file "taxi chicago.JPEG" di folder yang sama.
"""

# pip install streamlit pandas numpy plotly scikit-learn folium streamlit-folium
# pip install google-cloud-bigquery db-dtypes pyarrow scipy

import os
import base64
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Chicago Taxi Dashboard 2023",
    layout="wide",
    initial_sidebar_state="expanded",
)

C_ORANGE = "#E87F24"
C_YELLOW = "#FFC81E"
C_CREAM  = "#FEFDDF"
C_BLUE   = "#73A5CA"
C_DARK   = "#2C2C2C"
C_MID    = "#5A5A5A"
C_LIGHT  = "#F5F5EF"
PALETTE  = [C_ORANGE, C_BLUE, C_YELLOW, "#C0643A", "#4A7FA5", "#D4A820", "#8B5E3C"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Poppins', sans-serif !important;
    background-color: #FAFAF5;
}}
h1, h2, h3, h4 {{
    font-family: 'Poppins', sans-serif !important;
    color: {C_DARK};
}}
.metric-card {{
    background: {C_CREAM};
    border: 2px solid {C_ORANGE};
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    margin-bottom: 4px;
}}
.metric-card .val {{
    font-size: 1.75rem;
    font-weight: 700;
    color: {C_ORANGE};
    line-height: 1.1;
}}
.metric-card .lbl {{
    font-size: 0.70rem;
    color: {C_MID};
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}}
.insight-box {{
    background: {C_CREAM};
    border-left: 5px solid {C_ORANGE};
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    font-size: 0.87rem;
    color: {C_DARK};
    margin-top: 16px;
    line-height: 1.7;
}}
.section-header {{
    font-size: 1.2rem;
    font-weight: 700;
    color: {C_DARK};
    border-bottom: 3px solid {C_ORANGE};
    padding-bottom: 7px;
    margin: 28px 0 18px 0;
}}
.ml-card {{
    background: white;
    border: 1px solid #E8E8E0;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
}}
.stTabs [data-baseweb="tab"] {{
    font-weight: 600;
    font-size: 0.88rem;
    color: {C_MID};
    padding: 8px 20px;
}}
.stTabs [aria-selected="true"] {{
    color: {C_ORANGE} !important;
    border-bottom-color: {C_ORANGE} !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FUNGSI HELPER
# ─────────────────────────────────────────────
def chart_layout(fig, title=None):
    upd = dict(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Poppins", font_color=C_DARK,
        title_font_size=13, title_font_family="Poppins",
        title_font_color=C_DARK,
        margin=dict(t=45, b=30, l=10, r=10),
    )
    if title:
        upd["title"] = title
    fig.update_layout(**upd)
    fig.update_xaxes(showgrid=False, linecolor="#E0E0E0")
    fig.update_yaxes(gridcolor="#F0F0F0", linecolor="#E0E0E0")
    return fig


def insight_box(text: str):
    st.markdown(
        f'<div class="insight-box"><b>Insight:</b> {text}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Menghubungkan ke BigQuery...")
def load_from_bigquery(project_id: str, limit: int = 500_000) -> pd.DataFrame:
    from google.auth import default as google_auth_default
    from google.cloud import bigquery

    try:
        creds, _ = google_auth_default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        client = bigquery.Client(project=project_id, credentials=creds)
    except Exception:
        client = bigquery.Client(project=project_id)

    query = f"""
        SELECT
            unique_key, taxi_id, trip_start_timestamp, trip_end_timestamp,
            trip_seconds, trip_miles,
            pickup_community_area, dropoff_community_area,
            fare, tips, trip_total,
            payment_type, company,
            pickup_latitude, pickup_longitude,
            dropoff_latitude, dropoff_longitude
        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
        WHERE DATE(trip_start_timestamp) BETWEEN '2023-01-01' AND '2023-12-31'
          AND fare > 0 AND trip_miles > 0 AND trip_seconds > 0
          AND RAND() < 0.2
        ORDER BY trip_start_timestamp
        LIMIT {limit}
    """
    return client.query(query).to_dataframe()


@st.cache_data(show_spinner="Menyiapkan data demo...")
def load_demo(n: int = 100_000) -> pd.DataFrame:
    np.random.seed(42)
    w = np.array([2,1,1,1,1,2, 4,6,7,6,5,6, 7,6,5,5,6,8, 9,8,7,6,5,3], dtype=float)
    hours  = np.random.choice(range(24), n, p=w/w.sum())
    months = np.random.randint(1, 13, n)
    days   = np.random.randint(0, 7, n)

    trip_miles   = np.random.exponential(3.5, n).clip(0.1, 60)
    trip_seconds = (trip_miles * 180 + np.random.normal(0, 120, n)).clip(60, 7200)
    fare         = (trip_miles * 2.25 + 3.25 + np.random.normal(0, 2, n)).clip(3, 300)

    payment = np.random.choice(
        ["Credit Card","Cash","Prcard","Unknown","Mobile"],
        n, p=[0.37, 0.26, 0.22, 0.11, 0.04]
    )
    has_tip = np.where(payment == "Credit Card",
                       np.random.binomial(1, 0.72, n),
                       np.random.binomial(1, 0.08, n))
    tips = np.where(has_tip, fare * np.random.uniform(0.10, 0.25, n), 0).clip(0, 100)

    pickup_areas = np.random.choice(
        [8, 32, 76, 28, 6, 24, 33, 56, 21, 38],
        n, p=[0.18,0.15,0.13,0.11,0.10,0.09,0.08,0.07,0.05,0.04]
    )
    dropoff_areas = np.random.choice(
        [8, 32, 76, 28, 6, 24, 33, 56, 21, 38],
        n, p=[0.15,0.16,0.14,0.12,0.10,0.09,0.08,0.07,0.05,0.04]
    )
    companies = np.random.choice(
        ["Flash Cab","Taxi Affiliation Services","Sun Taxi","City Service","Globe Taxi","Yellow Cab"],
        n, p=[0.30,0.25,0.18,0.12,0.08,0.07]
    )

    lat_base = {8:41.900, 32:41.882, 76:41.978, 28:41.853, 6:41.795,
                24:41.876, 33:41.867, 56:41.838, 21:41.845, 38:41.912}
    lon_base = {8:-87.631, 32:-87.630, 76:-87.905, 28:-87.618, 6:-87.608,
                24:-87.654, 33:-87.667, 56:-87.621, 21:-87.625, 38:-87.679}

    pickup_lat = np.array([lat_base[a] + np.random.normal(0, 0.01) for a in pickup_areas])
    pickup_lon = np.array([lon_base[a] + np.random.normal(0, 0.01) for a in pickup_areas])

    ts = pd.Timestamp("2023-01-01")
    start_ts = [
        ts + pd.Timedelta(days=int((m-1)*30.5 + np.random.randint(0,28)),
                          hours=int(h), minutes=int(np.random.randint(0,60)))
        for m, h in zip(months, hours)
    ]

    return pd.DataFrame({
        "unique_key":             [f"trip_{i:07d}" for i in range(n)],
        "trip_start_timestamp":   start_ts,
        "trip_seconds":           trip_seconds,
        "trip_miles":             trip_miles,
        "fare":                   fare,
        "tips":                   tips,
        "trip_total":             fare + tips,
        "payment_type":           payment,
        "company":                companies,
        "pickup_community_area":  pickup_areas,
        "dropoff_community_area": dropoff_areas,
        "pickup_latitude":        pickup_lat,
        "pickup_longitude":       pickup_lon,
    })


@st.cache_data
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing sesuai notebook:
    1. Hapus duplikasi unique_key
    2. Imputasi median untuk numerik, 'None' untuk kategorik
    3. Filter outlier ekstrem (sanity check)
    4. Ekstraksi fitur waktu (hour, day_of_week, month, is_weekend)
    5. Fitur turunan: trip_minutes, speed_mph, tip_pct, has_tip
    6. Kategorisasi jarak dan waktu hari
    """
    df = df.copy()

    # 1. Hapus duplikasi
    df = df.drop_duplicates(subset=["unique_key"])

    # 2. Handle missing values
    for col in ["trip_seconds","trip_miles","fare","tips","trip_total"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in ["payment_type","company"]:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # 3. Filter outlier ekstrem
    df = df[
        df["trip_miles"].between(0.01, 100) &
        df["trip_seconds"].between(1, 7200) &
        df["fare"].between(0.01, 500) &
        (df["tips"] >= 0) & (df["tips"] < 200)
    ]
    if "trip_total" in df.columns:
        df = df[df["trip_total"].between(0.01, 500)]

    # 4. Parsing timestamp dan ekstraksi fitur waktu
    if "trip_start_timestamp" in df.columns:
        df["trip_start_timestamp"] = pd.to_datetime(
            df["trip_start_timestamp"], errors="coerce"
        )
        if df["trip_start_timestamp"].dt.tz is not None:
            df["trip_start_timestamp"] = df["trip_start_timestamp"].dt.tz_localize(None)
        df["hour"]        = df["trip_start_timestamp"].dt.hour
        df["month"]       = df["trip_start_timestamp"].dt.month
        df["day_of_week"] = df["trip_start_timestamp"].dt.dayofweek

    # 5. Label hari dan bulan (Indonesia)
    day_map   = {0:"Senin",1:"Selasa",2:"Rabu",3:"Kamis",4:"Jumat",5:"Sabtu",6:"Minggu"}
    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Mei",6:"Jun",
                 7:"Jul",8:"Agu",9:"Sep",10:"Okt",11:"Nov",12:"Des"}
    df["day_name"]   = df["day_of_week"].map(day_map)
    df["month_name"] = df["month"].map(month_map)
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

    # 6. Fitur turunan
    df["trip_minutes"] = (df["trip_seconds"] / 60).round(2)
    df["speed_mph"]    = np.where(
        df["trip_minutes"] > 0,
        (df["trip_miles"] / (df["trip_seconds"] / 3600)).round(2), np.nan
    )
    df["tip_pct"] = np.where(df["fare"] > 0, (df["tips"] / df["fare"] * 100).round(2), 0)
    df["has_tip"] = (df["tips"] > 0).astype(int)

    # 7. Kategorisasi jarak
    bins   = [0, 1, 3, 5, 10, float("inf")]
    labels = ["< 1 mil","1-3 mil","3-5 mil","5-10 mil","> 10 mil"]
    df["distance_cat"] = pd.cut(df["trip_miles"], bins=bins, labels=labels)

    # 8. Kategorisasi waktu hari
    def time_of_day(h):
        if   6  <= h < 12: return "Pagi (06-12)"
        elif 12 <= h < 17: return "Siang (12-17)"
        elif 17 <= h < 21: return "Sore (17-21)"
        elif 21 <= h < 24: return "Malam (21-24)"
        else:              return "Dini Hari (00-06)"

    df["time_of_day"] = df["hour"].apply(time_of_day)
    return df


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<div style='font-weight:700;font-size:1.05rem;color:{C_ORANGE};margin-bottom:4px;'>"
        "Chicago Taxi 2023</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    data_source = st.radio(
        "Sumber Data",
        ["Demo (Data Simulasi)", "BigQuery (Akun Google)"],
        index=0,
    )

    project_id = None
    bq_limit   = 500_000

    if data_source == "BigQuery (Akun Google)":
        st.markdown(
            f"<div style='font-size:0.78rem;color:{C_MID};margin-bottom:6px;'>"
            "Pastikan sudah login: <code>gcloud auth application-default login</code></div>",
            unsafe_allow_html=True,
        )
        project_id = st.text_input("Google Cloud Project ID", placeholder="my-project-id")
        bq_limit   = st.number_input("Batas Baris", 10_000, 1_000_000, 500_000, step=50_000)

    st.write("")
    st.markdown("**Filter**")

    months_all  = list(range(1, 13))
    month_names = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
    sel_months  = st.multiselect(
        "Bulan", options=months_all,
        format_func=lambda x: month_names[x-1],
        default=months_all,
    )

    all_payments = ["Credit Card","Cash","Prcard","Unknown","Mobile"]
    sel_payment  = st.multiselect("Metode Pembayaran", all_payments, default=all_payments)

    max_miles = st.slider("Jarak Maksimum (mil)", 1, 100, 30)

    st.write("")
    st.caption("Dataset: BigQuery Public Data\nchicago_taxi_trips — 2023")


# ─────────────────────────────────────────────
# LOAD & PREPROCESS
# ─────────────────────────────────────────────
if data_source == "BigQuery (Akun Google)":
    if not project_id:
        st.warning("Masukkan Google Cloud Project ID di sidebar untuk menggunakan BigQuery.")
        st.stop()
    try:
        with st.spinner("Mengambil data dari BigQuery..."):
            raw_df = load_from_bigquery(project_id, bq_limit)
    except Exception as e:
        st.error(
            f"Gagal terhubung ke BigQuery.\n\n"
            f"**Kemungkinan penyebab:**\n"
            f"Jalankan gcloud auth application-default login di terminal\n"
            f"Pastikan BigQuery API sudah diaktifkan di project\n"
            f"Pastikan Project ID sudah benar\n\n"
            f"**Detail error:** `{str(e)}`"
        )
        st.stop()
else:
    raw_df = load_demo()

df = preprocess(raw_df)

if sel_months:
    df = df[df["month"].isin(sel_months)]
if sel_payment:
    df = df[df["payment_type"].isin(sel_payment)]
df = df[df["trip_miles"] <= max_miles]

if len(df) == 0:
    st.warning("Tidak ada data sesuai filter. Sesuaikan pilihan filter di sidebar.")
    st.stop()


# ─────────────────────────────────────────────
# HEADER — foto jadi background penuh, judul di kiri
# ─────────────────────────────────────────────
FOTO_PATH = r"C:\Users\User\Downloads\taxi chicago.JPEG"

if os.path.exists(FOTO_PATH):
    import base64
    with open(FOTO_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <div style="
        background-image: url('data:image/jpeg;base64,{img_b64}');
        background-size: cover;
        background-position: center;
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 8px;
        height: 200px;
        display: flex;
        align-items: center;
        position: relative;
    ">
        <div style="
            position: absolute;
            inset: 0;
            background: linear-gradient(to right, rgba(0,0,0,0.70) 0%, rgba(0,0,0,0.40) 60%, rgba(0,0,0,0.10) 100%);
            border-radius: 16px;
        "></div>
        <div style="position: relative; z-index: 2; padding: 28px 32px;">
            <div style="font-family:'Poppins',sans-serif; font-size:2rem; font-weight:800; color:{C_ORANGE}; line-height:1.15;">
                Chicago Taxi Trips
            </div>
            <div style="font-family:'Poppins',sans-serif; font-size:1rem; font-weight:600; color:{C_YELLOW}; margin-top:4px;">
                Analisis Data Berbasis Cloud
            </div>
            <div style="font-family:'Poppins',sans-serif; font-size:0.82rem; color:rgba(255,255,255,0.85); margin-top:8px; line-height:1.6;">
                Periode: Januari &ndash; Desember 2023<br>
                Dataset: BigQuery Public Data &mdash; bigquery-public-data.chicago_taxi_trips
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div style="
        background: linear-gradient(120deg, {C_ORANGE}, {C_YELLOW});
        border-radius: 18px; padding: 36px 38px; margin-bottom: 14px;
        text-align: left;
    ">
        <div style="font-size:2rem;font-weight:800;color:white;line-height:1.2;">
            Chicago Taxi Trips
        </div>
        <div style="font-size:0.95rem;color:rgba(255,255,255,0.85);margin-top:6px;">
            Analisis Data Berbasis Cloud &mdash; Januari hingga Desember 2023
        </div>
        <div style="font-size:0.78rem;color:rgba(255,255,255,0.65);margin-top:4px;">
            Letakkan "taxi chicago.JPEG" di folder yang sama untuk menampilkan foto.
        </div>
    </div>
    """, unsafe_allow_html=True)

if data_source == "Demo (Data Simulasi)":
    st.info("Menampilkan data simulasi. Pilih BigQuery di sidebar untuk data nyata.")

st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────
total_trips   = len(df)
total_revenue = df["trip_total"].sum()
avg_fare      = df["fare"].mean()
avg_tip_pct   = df.loc[df["has_tip"] == 1, "tip_pct"].mean() if df["has_tip"].sum() > 0 else 0
avg_miles     = df["trip_miles"].mean()
tip_rate      = df["has_tip"].mean() * 100

c1, c2, c3, c4, c5, c6 = st.columns(6)
for col, val, lbl in [
    (c1, f"{total_trips:,}",           "Total Perjalanan"),
    (c2, f"${total_revenue/1e6:.1f}M", "Total Pendapatan"),
    (c3, f"${avg_fare:.2f}",           "Rata Rata Tarif"),
    (c4, f"{avg_tip_pct:.1f}%",        "Rata Rata Tip"),
    (c5, f"{avg_miles:.1f} mi",        "Rata Rata Jarak"),
    (c6, f"{tip_rate:.1f}%",           "Tingkat Pemberi Tip"),
]:
    col.markdown(
        f'<div class="metric-card"><div class="val">{val}</div>'
        f'<div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Pola Waktu",
    "Wilayah Aktif",
    "Tip dan Pembayaran",
    "Distribusi Jarak",
    "Machine Learning",
])


# ══════════════════════════════════════════════
# TAB 1 — POLA WAKTU
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Pola Permintaan Taksi Berdasarkan Waktu</div>',
                unsafe_allow_html=True)

    # Per jam dan per hari (sesuai notebook: axes[0,0] dan axes[0,1])
    col_a, col_b = st.columns(2)

    with col_a:
        hourly  = df.groupby("hour").size().reset_index(name="count")
        peak_h  = int(hourly.loc[hourly["count"].idxmax(), "hour"])
        fig = px.bar(hourly, x="hour", y="count",
                     color="count",
                     color_continuous_scale=[[0, C_CREAM],[1, C_BLUE]],
                     labels={"hour":"Jam","count":"Jumlah Perjalanan"})
        fig.update_layout(coloraxis_showscale=False)
        fig.add_vline(x=peak_h, line_dash="dash", line_color="red",
                      annotation_text=f"Puncak: {peak_h}:00",
                      annotation_font_color="red")
        chart_layout(fig, "Jumlah Perjalanan per Jam")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        day_order = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]
        daily = df.groupby("day_name").size().reset_index(name="count")
        daily["day_name"] = pd.Categorical(daily["day_name"], categories=day_order, ordered=True)
        daily = daily.sort_values("day_name")
        # Weekend berwarna berbeda seperti notebook
        daily["color"] = daily["day_name"].isin(["Sabtu","Minggu"]).map(
            {True: C_ORANGE, False: C_BLUE}
        )
        fig2 = go.Figure(go.Bar(
            x=daily["day_name"], y=daily["count"],
            marker_color=daily["color"].tolist(),
            showlegend=False,
        ))
        chart_layout(fig2, "Jumlah Perjalanan per Hari")
        fig2.update_layout(
            annotations=[
                dict(x=0.98, y=0.98, xref="paper", yref="paper", showarrow=False,
                     text="<span style='color:#E87F24'>■</span> Akhir Pekan  "
                          "<span style='color:#73A5CA'>■</span> Hari Kerja",
                     font=dict(size=10), bgcolor="white", borderpad=4)
            ]
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Tren bulanan (sesuai notebook: axes[1,0]) + tarif rata-rata
    month_map_full = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Mei",6:"Jun",
                      7:"Jul",8:"Agu",9:"Sep",10:"Okt",11:"Nov",12:"Des"}
    monthly = df.groupby("month").agg(
        trips=("unique_key","count"), avg_fare=("fare","mean")
    ).reset_index()
    monthly["bulan"] = monthly["month"].map(month_map_full)

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Bar(x=monthly["bulan"], y=monthly["trips"],
                          name="Jumlah Perjalanan", marker_color=C_BLUE,
                          marker_line_width=0), secondary_y=False)
    fig3.add_trace(go.Scatter(x=monthly["bulan"], y=monthly["avg_fare"],
                              name="Rata Rata Tarif ($)", mode="lines+markers",
                              line=dict(color=C_ORANGE, width=3),
                              marker=dict(size=8, color=C_ORANGE)), secondary_y=True)
    fig3.update_yaxes(title_text="Jumlah Perjalanan", secondary_y=False)
    fig3.update_yaxes(title_text="Rata Rata Tarif ($)", secondary_y=True)
    fig3.update_layout(legend=dict(orientation="h", y=-0.18))
    chart_layout(fig3, "Tren Bulanan: Perjalanan dan Tarif")
    st.plotly_chart(fig3, use_container_width=True)

    # Heatmap jam x hari (sesuai notebook: axes[1,1])
    pivot_hd = df.groupby(["day_name","hour"]).size().unstack(fill_value=0)
    pivot_hd = pivot_hd.reindex([d for d in day_order if d in pivot_hd.index])
    fig4 = px.imshow(pivot_hd,
                     color_continuous_scale=[[0,C_CREAM],[0.5,C_YELLOW],[1,C_ORANGE]],
                     labels=dict(x="Jam", y="Hari", color="Trips"),
                     aspect="auto")
    chart_layout(fig4, "Heatmap: Intensitas Perjalanan per Jam dan Hari")
    st.plotly_chart(fig4, use_container_width=True)

    low_month  = monthly.loc[monthly["trips"].idxmin(), "bulan"]
    high_month = monthly.loc[monthly["trips"].idxmax(), "bulan"]
    insight_box(
        f"Jam sibuk puncak terjadi pada <b>jam {peak_h}:00</b>, sesuai pola siang hingga sore hari. "
        f"Hari kerja mendominasi volume perjalanan dibanding akhir pekan (Sabtu-Minggu). "
        f"Secara bulanan, permintaan tertinggi terjadi pada <b>{high_month}</b> "
        f"dan terendah pada <b>{low_month}</b>. "
        f"Heatmap mengkonfirmasi bahwa kombinasi Jumat hingga Sabtu malam memiliki intensitas tertinggi "
        f"karena aktivitas hiburan dan pulang kerja."
    )


# ══════════════════════════════════════════════
# TAB 2 — WILAYAH AKTIF
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Wilayah dengan Aktivitas Taksi Tertinggi</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Top 20 pickup area (sesuai notebook)
        top_pickup = (df["pickup_community_area"].dropna()
                      .value_counts().head(20).reset_index())
        top_pickup.columns = ["area","count"]
        top_pickup["area"] = "Area " + top_pickup["area"].astype(int).astype(str)
        fig5 = px.bar(top_pickup.sort_values("count"), x="count", y="area",
                      orientation="h", color="count",
                      color_continuous_scale=[[0,C_CREAM],[1,C_BLUE]],
                      labels={"count":"Jumlah Pickup","area":"Area"})
        fig5.update_layout(coloraxis_showscale=False)
        chart_layout(fig5, "Top 20 Pickup Community Area")
        st.plotly_chart(fig5, use_container_width=True)

    with col_b:
        # Perbandingan Pickup vs Dropoff (sesuai notebook)
        pickup_vc  = df["pickup_community_area"].value_counts()
        dropoff_vc = df["dropoff_community_area"].value_counts() if "dropoff_community_area" in df.columns else pd.Series(dtype=float)
        compare_df = pd.DataFrame({"Pickup": pickup_vc, "Dropoff": dropoff_vc}).fillna(0)
        top_area   = compare_df.sum(axis=1).sort_values(ascending=False).head(10).index
        compare_df = compare_df.loc[top_area].reset_index()
        compare_df.columns = ["area","Pickup","Dropoff"]
        compare_df["area"] = "Area " + compare_df["area"].astype(int).astype(str)
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(name="Pickup", x=compare_df["area"], y=compare_df["Pickup"],
                              marker_color=C_ORANGE))
        fig6.add_trace(go.Bar(name="Dropoff", x=compare_df["area"], y=compare_df["Dropoff"],
                              marker_color=C_BLUE))
        fig6.update_layout(barmode="group", legend=dict(orientation="h", y=-0.2))
        chart_layout(fig6, "Perbandingan Pickup vs Dropoff per Wilayah")
        st.plotly_chart(fig6, use_container_width=True)

    # Heatmap area x jam (sesuai notebook)
    top_areas = df["pickup_community_area"].value_counts().head(10).index
    df_heat = df[df["pickup_community_area"].isin(top_areas)]
    pivot_ah = df_heat.pivot_table(
        index="pickup_community_area", columns="hour",
        values="unique_key", aggfunc="count", fill_value=0
    )
    pivot_ah.index = ["Area " + str(int(i)) for i in pivot_ah.index]
    fig_ah = px.imshow(pivot_ah,
                       color_continuous_scale=[[0,C_CREAM],[0.5,C_YELLOW],[1,C_ORANGE]],
                       labels=dict(x="Jam", y="Community Area", color="Trips"),
                       aspect="auto")
    chart_layout(fig_ah, "Kepadatan Aktivitas Berdasarkan Jam dan Wilayah")
    st.plotly_chart(fig_ah, use_container_width=True)

    # Peta
    st.markdown('<div class="section-header">Peta Persebaran Aktivitas Taksi</div>',
                unsafe_allow_html=True)

    has_coords = (
        "pickup_latitude" in df.columns and
        "pickup_longitude" in df.columns and
        df["pickup_latitude"].notna().sum() > 100
    )

    if has_coords:
        try:
            import folium
            from streamlit_folium import st_folium

            df_map = df.dropna(subset=["pickup_latitude","pickup_longitude"])
            df_map = df_map[
                df_map["pickup_latitude"].between(41.6, 42.1) &
                df_map["pickup_longitude"].between(-87.9, -87.5)
            ]

            map_mode = st.radio(
                "Mode Peta", ["Heatmap Aktivitas", "Scatter per Area"],
                horizontal=True
            )

            m = folium.Map(location=[41.8781, -87.6298], zoom_start=11,
                           tiles="CartoDB dark_matter")

            if map_mode == "Heatmap Aktivitas":
                from folium.plugins import HeatMap
                sample_map = df_map.sample(min(30000, len(df_map)), random_state=42)
                heat_data  = [[r["pickup_latitude"], r["pickup_longitude"]]
                              for _, r in sample_map.iterrows()]
                HeatMap(heat_data, radius=8, blur=10, min_opacity=0.4).add_to(m)
            else:
                area_coords = df_map.groupby("pickup_community_area").agg(
                    lat=("pickup_latitude","mean"),
                    lon=("pickup_longitude","mean"),
                    count=("unique_key","count")
                ).reset_index().dropna()
                max_count = area_coords["count"].max()
                for _, row in area_coords.iterrows():
                    radius = max(5, int(row["count"] / max_count * 30))
                    folium.CircleMarker(
                        location=[row["lat"], row["lon"]],
                        radius=radius, color=C_ORANGE,
                        fill=True, fill_color=C_ORANGE, fill_opacity=0.6,
                        tooltip=f"Area {int(row['pickup_community_area'])}: {int(row['count']):,} trips"
                    ).add_to(m)

            st_folium(m, width=None, height=450, returned_objects=[])

        except ImportError:
            area_coord_map = {
                8:  (41.900,-87.631), 32:(41.882,-87.630), 76:(41.978,-87.905),
                28: (41.853,-87.618), 6: (41.795,-87.608), 24:(41.876,-87.654),
                33: (41.867,-87.667), 56:(41.838,-87.621), 21:(41.845,-87.625),
                38: (41.912,-87.679)
            }
            area_count = df["pickup_community_area"].value_counts().reset_index()
            area_count.columns = ["area","count"]
            area_count["lat"] = area_count["area"].map(lambda x: area_coord_map.get(x,(41.88,-87.63))[0])
            area_count["lon"] = area_count["area"].map(lambda x: area_coord_map.get(x,(41.88,-87.63))[1])
            area_count["label"] = "Area " + area_count["area"].astype(int).astype(str)
            fig_mb = px.scatter_mapbox(
                area_count, lat="lat", lon="lon", size="count",
                hover_name="label", hover_data={"count":True,"lat":False,"lon":False},
                color="count", color_continuous_scale=[[0,C_YELLOW],[1,C_ORANGE]],
                mapbox_style="carto-darkmatter", zoom=10, height=450, size_max=40
            )
            fig_mb.update_layout(margin=dict(t=0,b=0,l=0,r=0), coloraxis_showscale=False)
            st.plotly_chart(fig_mb, use_container_width=True)
    else:
        st.info("Data koordinat tidak tersedia pada mode Demo. Gunakan BigQuery untuk peta interaktif.")
        area_coord_map = {
            8:  (41.900,-87.631), 32:(41.882,-87.630), 76:(41.978,-87.905),
            28: (41.853,-87.618), 6: (41.795,-87.608), 24:(41.876,-87.654),
            33: (41.867,-87.667), 56:(41.838,-87.621), 21:(41.845,-87.625),
            38: (41.912,-87.679)
        }
        area_count = df["pickup_community_area"].value_counts().reset_index()
        area_count.columns = ["area","count"]
        area_count["lat"] = area_count["area"].map(lambda x: area_coord_map.get(x,(41.88,-87.63))[0])
        area_count["lon"] = area_count["area"].map(lambda x: area_coord_map.get(x,(41.88,-87.63))[1])
        area_count["label"] = "Area " + area_count["area"].astype(int).astype(str)
        fig_mb = px.scatter_mapbox(
            area_count, lat="lat", lon="lon", size="count",
            hover_name="label", hover_data={"count":True,"lat":False,"lon":False},
            color="count", color_continuous_scale=[[0,C_YELLOW],[1,C_ORANGE]],
            mapbox_style="carto-darkmatter", zoom=10, height=450, size_max=40
        )
        fig_mb.update_layout(margin=dict(t=0,b=0,l=0,r=0), coloraxis_showscale=False)
        st.plotly_chart(fig_mb, use_container_width=True)

    top_area_id = int(df["pickup_community_area"].dropna().value_counts().idxmax())
    insight_box(
        f"Community Area <b>{top_area_id}</b> mencatat volume pickup tertinggi, "
        f"berkorelasi dengan kawasan Loop dan Near North Side — pusat bisnis Chicago. "
        f"Bandara O'Hare (Area 76) menjadi titik aktivitas terbesar kedua, "
        f"didominasi perjalanan jarak jauh dengan tarif di atas rata rata. "
        f"Wilayah pinggiran kota menunjukkan aktivitas yang jauh lebih rendah dan sporadis."
    )


# ══════════════════════════════════════════════
# TAB 3 — TIP & PEMBAYARAN
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Analisis Tip dan Metode Pembayaran</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        tip_by_pay = df.groupby("payment_type").agg(
            tip_rate=("has_tip","mean"),
            avg_tip=("tip_pct","mean"),
            count=("unique_key","count")
        ).reset_index()
        tip_by_pay["tip_rate_pct"] = tip_by_pay["tip_rate"] * 100
        fig9 = px.bar(tip_by_pay.sort_values("tip_rate_pct", ascending=False),
                      x="payment_type", y="tip_rate_pct",
                      color="tip_rate_pct",
                      color_continuous_scale=[[0,C_CREAM],[1,C_ORANGE]],
                      labels={"payment_type":"Metode","tip_rate_pct":"% Memberi Tip"})
        fig9.update_layout(coloraxis_showscale=False)
        chart_layout(fig9, "Tingkat Pemberian Tip per Metode Pembayaran")
        st.plotly_chart(fig9, use_container_width=True)

    with col_b:
        pay_dist = df["payment_type"].value_counts().reset_index()
        pay_dist.columns = ["payment_type","count"]
        fig10 = px.pie(pay_dist, names="payment_type", values="count",
                       color_discrete_sequence=PALETTE)
        fig10.update_traces(textposition="inside", textinfo="percent+label",
                            textfont_family="Poppins")
        chart_layout(fig10, "Distribusi Metode Pembayaran")
        fig10.update_layout(showlegend=False)
        st.plotly_chart(fig10, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        # Rata rata tip per metode (sesuai notebook viz_03_tip_analysis)
        tip_pay_mean = df.groupby("payment_type")["tips"].mean().sort_values(ascending=False).reset_index()
        colors_bars  = [C_BLUE] * len(tip_pay_mean)
        colors_bars[0] = C_ORANGE  # highlight tertinggi
        fig_tp = go.Figure(go.Bar(
            x=tip_pay_mean["payment_type"], y=tip_pay_mean["tips"],
            marker_color=colors_bars,
            text=[f"${v:.2f}" for v in tip_pay_mean["tips"]],
            textposition="outside"
        ))
        chart_layout(fig_tp, "Rata Rata Tip per Metode Pembayaran")
        st.plotly_chart(fig_tp, use_container_width=True)

    with col_d:
        # % tip per kategori jarak (sesuai notebook viz_03_tip_analysis)
        tip_by_dist = df.groupby("distance_cat", observed=True)["tip_pct"].mean().reset_index()
        fig_td = px.bar(tip_by_dist, x="distance_cat", y="tip_pct",
                        color="tip_pct",
                        color_continuous_scale=[[0,C_CREAM],[1,C_BLUE]],
                        text=tip_by_dist["tip_pct"].apply(lambda v: f"{v:.1f}%"),
                        labels={"distance_cat":"Kategori Jarak","tip_pct":"% Tip dari Fare"})
        fig_td.update_traces(textposition="outside")
        fig_td.update_layout(coloraxis_showscale=False)
        chart_layout(fig_td, "Rata Rata Persen Tip per Kategori Jarak")
        st.plotly_chart(fig_td, use_container_width=True)

    # Scatter jarak vs tip
    sample = df[df["has_tip"]==1].sample(min(3000, df["has_tip"].sum()), random_state=42)
    fig12 = px.scatter(sample, x="trip_miles", y="tip_pct",
                       color="payment_type", opacity=0.5,
                       labels={"trip_miles":"Jarak (mil)","tip_pct":"Tip (%)"},
                       color_discrete_sequence=PALETTE)
    chart_layout(fig12, "Hubungan Jarak vs Tip (%) per Metode Pembayaran")
    st.plotly_chart(fig12, use_container_width=True)

    best_pay  = tip_by_pay.loc[tip_by_pay["tip_rate_pct"].idxmax(), "payment_type"]
    worst_pay = tip_by_pay.loc[tip_by_pay["tip_rate_pct"].idxmin(), "payment_type"]
    insight_box(
        f"<b>{best_pay}</b> memiliki tingkat pemberian tip tertinggi, sementara "
        f"<b>{worst_pay}</b> hampir selalu menghasilkan tip nol karena tidak tercatat secara digital. "
        f"Fitur paling penting dalam model adalah <b>fare</b> dan <b>trip_miles</b>, "
        f"menunjukkan bahwa nominal perjalanan lebih menentukan tip dibanding waktu atau lokasi. "
        f"Perjalanan jarak menengah (3 hingga 10 mil) cenderung memiliki persentase tip tertinggi."
    )


# ══════════════════════════════════════════════
# TAB 4 — DISTRIBUSI JARAK
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Distribusi dan Karakteristik Jarak Perjalanan</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        mean_mi   = df["trip_miles"].mean()
        median_mi = df["trip_miles"].median()
        fig13 = px.histogram(df[df["trip_miles"] <= 30], x="trip_miles",
                             nbins=80, color_discrete_sequence=[C_BLUE],
                             labels={"trip_miles":"Jarak (mil)"})
        fig13.add_vline(x=mean_mi, line_dash="dash", line_color=C_ORANGE,
                        annotation_text=f"Mean: {mean_mi:.2f} mi",
                        annotation_font_color=C_ORANGE)
        fig13.add_vline(x=median_mi, line_dash="dash", line_color=C_YELLOW,
                        annotation_text=f"Median: {median_mi:.2f} mi",
                        annotation_font_color=C_YELLOW)
        chart_layout(fig13, "Distribusi Jarak Perjalanan")
        st.plotly_chart(fig13, use_container_width=True)

    with col_b:
        dist_counts = df["distance_cat"].value_counts().reset_index()
        dist_counts.columns = ["distance_cat","count"]
        fig14 = px.pie(dist_counts, names="distance_cat", values="count",
                       color_discrete_sequence=[C_BLUE,C_ORANGE,C_YELLOW,"#1a5276","#aed6f1"])
        fig14.update_traces(textposition="inside", textinfo="percent+label",
                            textfont_family="Poppins",
                            marker=dict(line=dict(color="white",width=1.5)))
        chart_layout(fig14, "Proporsi Kategori Jarak")
        fig14.update_layout(showlegend=False)
        st.plotly_chart(fig14, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        fig15 = px.bar(dist_counts.sort_values("count", ascending=False),
                       x="distance_cat", y="count",
                       color="count",
                       color_continuous_scale=[[0,C_CREAM],[1,C_BLUE]],
                       labels={"distance_cat":"Kategori Jarak","count":"Jumlah Perjalanan"})
        fig15.update_layout(coloraxis_showscale=False)
        chart_layout(fig15, "Jumlah Perjalanan per Kategori Jarak")
        st.plotly_chart(fig15, use_container_width=True)

    with col_d:
        # Scatter jarak vs tarif dengan tren linear (sesuai notebook)
        sample2 = df.sample(min(5000, len(df)), random_state=42)
        s2 = sample2[sample2["trip_miles"] <= 25]
        fig16 = px.scatter(s2, x="trip_miles", y="fare",
                           color="distance_cat", opacity=0.4,
                           trendline="ols",
                           labels={"trip_miles":"Jarak (mil)","fare":"Tarif ($)"},
                           color_discrete_sequence=[C_BLUE,C_ORANGE,C_YELLOW,"#1a5276","#aed6f1"])
        chart_layout(fig16, "Hubungan Jarak vs Tarif")
        st.plotly_chart(fig16, use_container_width=True)

    # Statistik per kategori jarak
    st.markdown("#### Statistik per Kategori Jarak")
    stats = df.groupby("distance_cat", observed=True).agg(
        Jumlah=("unique_key","count"),
        Jarak_Rata=("trip_miles","mean"),
        Tarif_Rata=("fare","mean"),
        Tip_Rata=("tip_pct","mean"),
        Durasi_Menit=("trip_minutes","mean"),
    ).round(2).reset_index()
    stats.columns = ["Kategori","Jumlah","Jarak Rata Rata (mi)",
                     "Tarif Rata Rata ($)","Tip Rata Rata (%)","Durasi Rata Rata (mnt)"]
    st.dataframe(stats, use_container_width=True, hide_index=True)

    pct_short = df["distance_cat"].isin(["< 1 mil","1-3 mil"]).sum() / len(df) * 100
    insight_box(
        f"Distribusi jarak sangat <i>right-skewed</i> — <b>{pct_short:.1f}%</b> perjalanan "
        f"berjarak kurang dari 3 mil, mengkonfirmasi dominasi perjalanan pendek perkotaan. "
        f"Median jarak ({median_mi:.2f} mil) jauh lebih kecil dari mean ({mean_mi:.2f} mil), "
        f"menunjukkan pengaruh outlier perjalanan jauh (rute bandara). "
        f"Korelasi jarak dan tarif bersifat linear kuat, mengindikasikan struktur tarif yang transparan dan proporsional."
    )


# ══════════════════════════════════════════════
# TAB 5 — MACHINE LEARNING (FIXED)
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Machine Learning Insights</div>',
                unsafe_allow_html=True)

    ml_tab1, ml_tab2 = st.tabs([
        "Clustering Wilayah (Agglomerative + PCA)",
        "Random Forest — Prediksi Tip"
    ])

    # ─────────────────────────────
    # CLUSTERING (FIXED STABLE VERSION)
    # ─────────────────────────────
    with ml_tab1:
        st.markdown("#### Clustering Wilayah Berdasarkan Aktivitas")

        n_clusters = st.slider("Jumlah Cluster (K)", 2, 6, 3)

        with st.spinner("Menjalankan clustering..."):

            area_cluster = (
                df.groupby("pickup_community_area")
                .agg(
                    trip_count=("unique_key", "count"),
                    avg_distance=("trip_miles", "mean"),
                    avg_duration=("trip_seconds", "mean"),
                    avg_fare=("fare", "mean"),
                    avg_tip=("tips", "mean"),
                )
                .reset_index()
                .dropna()
            )

            # log transform
            area_cluster["trip_count_log"] = np.log1p(area_cluster["trip_count"])

            X = area_cluster[
                ["trip_count_log", "avg_distance", "avg_duration", "avg_fare", "avg_tip"]
            ]

            # scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            area_cluster["PC1"] = X_pca[:, 0]
            area_cluster["PC2"] = X_pca[:, 1]

            var_explained = np.sum(pca.explained_variance_ratio_) * 100

            # clustering
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X_pca)

            area_cluster["cluster"] = labels.astype(str)

            # safe silhouette
            silhouette = (
                silhouette_score(X_pca, labels)
                if len(set(labels)) > 1 else 0
            )

        # ── VISUAL (FIXED: NO RAW ARRAY IN PLOTLY) ──
        fig = px.scatter(
            area_cluster,
            x="PC1",
            y="PC2",
            color="cluster",
            hover_name="pickup_community_area",
            color_discrete_sequence=PALETTE
        )

        chart_layout(fig, f"Agglomerative Clustering (PCA Variance: {var_explained:.1f}%)")
        st.plotly_chart(fig, use_container_width=True)

        # ── PROFILE ──
        profile = area_cluster.groupby("cluster").agg(
            total_area=("pickup_community_area", "count"),
            total_trips=("trip_count", "sum"),
            avg_fare=("avg_fare", "mean"),
            avg_distance=("avg_distance", "mean"),
            avg_tip=("avg_tip", "mean"),
        ).reset_index()

        st.markdown("#### Profil Cluster")
        st.dataframe(profile, use_container_width=True, hide_index=True)

        insight_box(
            f"Clustering menghasilkan <b>{n_clusters} cluster</b> "
            f"dengan silhouette score <b>{silhouette:.4f}</b>. "
            f"PCA menjelaskan <b>{var_explained:.1f}% variansi</b>."
        )

    # ─────────────────────────────
    # RANDOM FOREST (FIXED SAFETY)
    # ─────────────────────────────
    with ml_tab2:
        st.markdown("#### Random Forest — Faktor yang Mempengaruhi Tip")

        with st.spinner("Melatih model..."):

            features = [
                "trip_miles", "trip_minutes", "fare", "payment_type",
                "hour", "day_of_week", "is_weekend"
            ]

            df_rf = df[features + ["tips"]].copy()

            # drop NA biar aman
            df_rf = df_rf.dropna()

            # encode categorical
            le = LabelEncoder()
            df_rf["payment_type"] = le.fit_transform(df_rf["payment_type"].astype(str))

            X = df_rf[features]
            y = df_rf["tips"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            rf.fit(X_train, y_train)
            pred = rf.predict(X_test)

            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)

        # metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.3f}")
        c2.metric("R2 Score", f"{r2:.3f}")
        c3.metric("Explained Variance", f"{r2*100:.1f}%")

        # feature importance
        importance = pd.DataFrame({
            "feature": features,
            "importance": rf.feature_importances_
        }).sort_values("importance")

        fig_imp = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues"
        )

        chart_layout(fig_imp, "Feature Importance")
        st.plotly_chart(fig_imp, use_container_width=True)

        # actual vs pred (FIXED SAFE MAX)
        max_val = max(max(y_test), max(pred))

        fig_pred = px.scatter(
            x=y_test,
            y=pred,
            labels={"x": "Actual Tip", "y": "Predicted Tip"},
            opacity=0.5
        )

        fig_pred.add_shape(
            type="line",
            x0=0, y0=0,
            x1=max_val, y1=max_val,
            line=dict(color="red", dash="dash")
        )

        chart_layout(fig_pred, "Actual vs Predicted Tip")
        st.plotly_chart(fig_pred, use_container_width=True)

        insight_box(
            f"Model Random Forest menghasilkan R² <b>{r2:.3f}</b>. "
            f"Fitur paling berpengaruh adalah <b>{importance.iloc[-1]['feature']}</b>."
        )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.write("")
st.markdown(
    f"<div style='text-align:center;color:{C_MID};font-size:0.78rem;padding:10px 0;"
    f"border-top:1px solid #E0E0D8;margin-top:8px;'>"
    "Dashboard Chicago Taxi Trips 2023 &nbsp;|&nbsp; "
    "Data: BigQuery Public Data &nbsp;|&nbsp; "
    "Dibuat dengan Streamlit dan Plotly"
    "</div>",
    unsafe_allow_html=True,
)