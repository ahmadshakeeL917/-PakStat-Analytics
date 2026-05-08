import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="PakStat | Road Accident Analysis",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stHeader"], [data-testid="stToolbar"] {
      background-color: #0d0d0d !important;
      color: #e0e0e0 !important;
  }
  [data-testid="stSidebar"] {
      background-color: #111111 !important;
      border-right: 1px solid #222;
  }
  [data-testid="stSidebar"] * { color: #aaaaaa !important; }
  .block-container { padding-top: 1.2rem !important; }

  .logo-bar { padding: 20px 20px 4px; }
  .logo-text { font-size: 1.55rem; font-weight: 900; letter-spacing: -1px; }
  .logo-pak  { color: #ffffff; }
  .logo-stat { color: #e50914; }
  .logo-sub  { font-size: 0.65rem; letter-spacing: 3.5px; color: #444444;
               text-transform: uppercase; padding: 2px 20px 18px; }
  .nav-label { font-size: 0.62rem; letter-spacing: 2.5px; color: #444444;
               text-transform: uppercase; padding: 16px 20px 6px; }

  .page-title { font-size: 1.75rem; font-weight: 800; color: #ffffff; margin-bottom: 4px; }
  .page-sub   { font-size: 0.86rem; color: #444444; margin-bottom: 1rem; }
  .divider    { border: none; border-top: 1px solid #1e1e1e; margin: 0.4rem 0 1.2rem; }

  .kpi-card {
      background: #141414;
      border: 1px solid #1e1e1e;
      border-top: 2px solid #e50914;
      border-radius: 10px; padding: 16px 18px;
  }
  .kpi-label { font-size: 0.70rem; color: #444444; text-transform: uppercase;
               letter-spacing: 1.5px; margin-bottom: 6px; }
  .kpi-value { font-size: 1.85rem; font-weight: 800; color: #ffffff; }
  .kpi-delta { font-size: 0.76rem; color: #e50914; margin-top: 4px; }

  .sec-hdr { font-size: 0.72rem; color: #444444; text-transform: uppercase;
             letter-spacing: 2px; margin: 1.4rem 0 0.6rem;
             border-bottom: 1px solid #1e0035; padding-bottom: 5px; }

  .insight { background: #141414; border: 1px solid #1e1e1e;
             border-left: 3px solid #e50914;
             padding: 10px 14px; border-radius: 6px;
             font-size: 0.84rem; color: #aaaaaa; margin: 10px 0; }

  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  div[role="radiogroup"] label {
      padding: 9px 20px !important;
      font-size: 0.90rem !important;
      font-weight: 500 !important;
      color: #666666 !important;
      border-left: 3px solid transparent !important;
      border-radius: 0 !important;
      display: block !important;
  }
  div[role="radiogroup"] label:hover { color: #ffffff !important; }

  .stSlider > div > div { background: #1e1e1e !important; }
  .stSelectbox > div > div { background: #141414 !important; color: #e0e0e0 !important; }
  .stMultiSelect > div > div { background: #141414 !important; color: #e0e0e0 !important; }
  .stButton > button {
      background: #1a0000 !important; color: #e50914 !important;
      border: 1px solid #e50914 !important; border-radius: 8px !important;
  }
  .stButton > button:hover { background: #1e1e1e !important; }
  .stDataFrame { background: #141414 !important; }
</style>
""", unsafe_allow_html=True)

PURPLE = "#e50914"
DARK   = dict(
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#111111",
    font=dict(color="#aaaaaa", family="Arial"),
    xaxis=dict(gridcolor="#1e0035", linecolor="#1e1e1e", zerolinecolor="#1e1e1e"),
    yaxis=dict(gridcolor="#1e0035", linecolor="#1e1e1e", zerolinecolor="#1e1e1e"),
    legend=dict(bgcolor="#141414", bordercolor="#1e1e1e", borderwidth=1),
    margin=dict(t=45, b=40, l=50, r=20),
)
COLORS = ["#e50914","#ff6b6b","#ffa94d","#ff6b6b","#ffa94d","#ffe066","#69db7c","#4dabf7"]
SCALE  = [[0,"#0d0d0d"],[0.3,"#2a0000"],[0.7,"#990000"],[1,"#e50914"]]

@st.cache_data
def load_data():
    df = pd.read_csv("RTA.csv")
    veh_cols = ["BicycleInvovled","BikesInvolved","BusesInvolved","CarsInvolved",
                "CartInvovled","RickshawsInvolved","TractorInvovled","TrainsInvovled",
                "TrucksInvolved","VansInvolved","OthersInvolved"]
    df["Vehicles_Involved"] = df[veh_cols].sum(axis=1).clip(lower=1)
    def dominant_vehicle(row):
        mapping = {"BicycleInvovled":"Bicycle","BikesInvolved":"Motorcycle",
                   "BusesInvolved":"Bus","CarsInvolved":"Car","CartInvovled":"Cart",
                   "RickshawsInvolved":"Rickshaw","TractorInvovled":"Tractor",
                   "TrainsInvovled":"Train","TrucksInvolved":"Truck",
                   "VansInvolved":"Van","OthersInvolved":"Other"}
        vals = {mapping[c]: row[c] for c in veh_cols}
        best = max(vals, key=vals.get)
        return best if vals[best] > 0 else "Other"
    df["Vehicle_Type"] = df.apply(dominant_vehicle, axis=1)
    df["Is_Fatal"] = (df["ACCLASS"] == "Fatal").astype(int)
    sev_map = {"Minor":"Minor Injury","Single Fracture":"Serious Injury",
               "Multiple Fractures":"Serious Injury","Head Injury":"Fatal-Moderate",
               "Spinal Injury":"Fatal-Severe"}
    df["Severity"] = df["InjuryType"].map(sev_map).fillna("Minor Injury")
    df.loc[df["ACCLASS"] == "Fatal", "Severity"] = "Fatal-Critical"
    df.rename(columns={"TotalPatientsInEmergency":"Patients",
                       "Nature of Weekday":"Weekday_Type",
                       "Age Range":"Age_Range"}, inplace=True)
    time_map = {"0 to 6":3,"6 to 12":9,"12 to 18":15,"18 to 24":21}
    df["Hour"] = df["Time"].map(time_map).fillna(12).astype(int)
    df["Month_Name"] = df["Month"].map({1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                                         7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
    df["Cause"] = df["Cause"].str.strip()
    return df

df = load_data()

with st.sidebar:
    st.markdown("""
    <div class="logo-bar">
      <span class="logo-text">
        <span class="logo-pak">Pak</span><span class="logo-stat">Stat</span>
      </span>
    </div>
    <div class="logo-sub">Analytics Studio</div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Pages</div>', unsafe_allow_html=True)
    pages = ["Home","Channel Overview","Descriptive Statistics","Growth Analysis",
             "Probability Analysis","Prediction Model","Best Time to Travel","Severity Explorer"]
    page = st.radio("", pages, label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="nav-label">Filters</div>', unsafe_allow_html=True)
    yr_min, yr_max = int(df["Year"].min()), int(df["Year"].max())
    yr_range = st.slider("Year Range", yr_min, yr_max, (yr_min, yr_max))
    all_weather = sorted(df["Weather"].unique())
    sel_weather = st.multiselect("Weather", all_weather, default=all_weather)
    all_cause = sorted(df["Cause"].unique())
    sel_cause = st.multiselect("Cause", all_cause, default=all_cause)
    st.markdown("---")
    st.markdown('<div style="font-size:0.70rem;color:#444444;padding:6px 0;">Source: Kaggle — RTA Dataset Pakistan<br>Records: 46,187 | Variables: 31<br>Period: 2020-2023</div>', unsafe_allow_html=True)

mask = (
    df["Year"].between(*yr_range) &
    df["Weather"].isin(sel_weather if sel_weather else all_weather) &
    df["Cause"].isin(sel_cause if sel_cause else all_cause)
)
d = df[mask].copy()

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;min-height:78vh;text-align:center;">
        <div style="font-size:5rem;font-weight:900;letter-spacing:-3px;line-height:1;">
            <span style="color:#ffffff;">Pak</span><span style="color:#e50914;">Stat</span>
        </div>
        <div style="width:120px;height:2px;background:linear-gradient(90deg,transparent,#e50914,transparent);margin:20px auto;"></div>
        <div style="font-size:0.88rem;color:#444444;letter-spacing:4px;text-transform:uppercase;">
            Advanced Road Accident Analytics &mdash; powered by real data.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CHANNEL OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Channel Overview":
    st.markdown('<div class="page-title">Channel Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Pakistan Road Traffic Accident Statistics — 2020 to 2023</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    total    = len(d)
    fatal    = int(d["Is_Fatal"].sum())
    nonfatal = total - fatal
    fatal_pct= fatal/total*100 if total else 0
    avg_pat  = d["Patients"].mean()
    avg_resp = d["responsetime"].mean()

    k1,k2,k3,k4,k5 = st.columns(5)
    kpis = [
        (f"{total:,}",        "Total Accidents",     f"{d['Year'].nunique()} years of data"),
        (f"{fatal:,}",        "Fatal Accidents",     f"{fatal_pct:.1f}% of total"),
        (f"{nonfatal:,}",     "Non-Fatal",           f"{100-fatal_pct:.1f}% of total"),
        (f"{avg_pat:.2f}",    "Avg Patients",        "Per emergency call"),
        (f"{avg_resp:.1f}m",  "Avg Response Time",   "Emergency response"),
    ]
    for col,(val,lbl,delta) in zip([k1,k2,k3,k4,k5],kpis):
        col.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">{lbl}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Accident Distribution</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        cause_df = d["Cause"].value_counts().reset_index()
        cause_df.columns = ["Cause","Count"]
        fig = px.bar(cause_df, x="Count", y="Cause", orientation="h",
                     color="Count", color_continuous_scale=SCALE, title="Accidents by Cause")
        fig.update_layout(**DARK, height=360, coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        acc_type = d["ACCLASS"].value_counts().reset_index()
        acc_type.columns = ["Type","Count"]
        fig = px.pie(acc_type, names="Type", values="Count", title="Fatal vs Non-Fatal",
                     hole=0.55, color_discrete_sequence=[PURPLE,"#2a0000"])
        fig.update_layout(**DARK, height=360)
        fig.update_traces(textfont_color="#ffffff", textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        wx_df = d["Weather"].value_counts().reset_index()
        wx_df.columns = ["Weather","Count"]
        fig = px.bar(wx_df, x="Weather", y="Count",
                     color="Count", color_continuous_scale=SCALE, title="Accidents by Weather")
        fig.update_layout(**DARK, height=340, coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        veh_df = d["Vehicle_Type"].value_counts().reset_index()
        veh_df.columns = ["Vehicle","Count"]
        fig = px.bar(veh_df, x="Vehicle", y="Count",
                     color="Count", color_continuous_scale=SCALE, title="Accidents by Vehicle Type")
        fig.update_layout(**DARK, height=340, coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-hdr">Injury Type and Light Condition</div>', unsafe_allow_html=True)
    c5,c6 = st.columns(2)
    with c5:
        inj_df = d["InjuryType"].value_counts().reset_index()
        inj_df.columns = ["Injury","Count"]
        fig = px.pie(inj_df, names="Injury", values="Count", title="Injury Type Distribution",
                     hole=0.45, color_discrete_sequence=COLORS)
        fig.update_layout(**DARK, height=320)
        fig.update_traces(textfont_color="#ffffff")
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        light_df = d["Light"].value_counts().reset_index()
        light_df.columns = ["Light","Count"]
        fig = px.bar(light_df, x="Light", y="Count",
                     color="Count", color_continuous_scale=SCALE, title="Accidents by Light Condition")
        fig.update_layout(**DARK, height=320, coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Descriptive Statistics":
    st.markdown('<div class="page-title">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Central Tendency, Spread, Shape and Confidence Intervals</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    num_cols = ["Patients","responsetime","Vehicles_Involved","Is_Fatal"]

    st.markdown('<div class="sec-hdr">Full Statistical Summary</div>', unsafe_allow_html=True)
    desc = d[num_cols].describe().T
    desc["Skewness"] = d[num_cols].skew()
    desc["Kurtosis"] = d[num_cols].kurtosis()
    desc["Variance"] = d[num_cols].var()
    desc = desc[["count","mean","std","Variance","min","25%","50%","75%","max","Skewness","Kurtosis"]]
    desc.columns = ["N","Mean","Std Dev","Variance","Min","Q1","Median","Q3","Max","Skewness","Kurtosis"]
    st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-hdr">Distribution — Histogram + KDE</div>', unsafe_allow_html=True)
        sel = st.selectbox("Variable", num_cols)
        vals = d[sel].dropna()
        mu = vals.mean()
        x_kde = np.linspace(vals.min(), vals.max(), 300)
        kde = stats.gaussian_kde(vals)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=vals, nbinsx=30, name="Observed",
                                   marker_color=PURPLE, opacity=0.55, histnorm="probability density"))
        fig.add_trace(go.Scatter(x=x_kde, y=kde(x_kde), mode="lines",
                                 name="KDE", line=dict(color="#ffffff", width=2)))
        fig.add_vline(x=mu, line_dash="dash", line_color="#ffa94d",
                      annotation_text=f"Mean={mu:.3f}", annotation_font_color="#ffa94d")
        fig.add_vline(x=vals.median(), line_dash="dot", line_color="#69db7c",
                      annotation_text=f"Median={vals.median():.3f}", annotation_font_color="#69db7c")
        fig.update_layout(**DARK, title=f"Distribution — {sel}", height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="sec-hdr">Box Plot by Season</div>', unsafe_allow_html=True)
        sel2 = st.selectbox("Variable for Box Plot", num_cols, index=0)
        fig = px.box(d, x="Season", y=sel2, color="Season",
                     color_discrete_sequence=COLORS, title=f"{sel2} by Season")
        fig.update_layout(**DARK, height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-hdr">95% Confidence Intervals</div>', unsafe_allow_html=True)
    ci_rows = []
    for col in num_cols:
        vals = d[col].dropna()
        n_,m_ = len(vals), vals.mean()
        ci = stats.t.interval(0.95, df=n_-1, loc=m_, scale=stats.sem(vals))
        ci_rows.append({"Variable":col,"N":n_,"Mean":round(m_,4),"Std Dev":round(vals.std(),4),
                        "95% CI Lower":round(ci[0],4),"95% CI Upper":round(ci[1],4),
                        "Margin of Error":round(ci[1]-m_,5)})
    ci_df = pd.DataFrame(ci_rows)
    st.dataframe(ci_df.style.format({"Mean":"{:.4f}","Std Dev":"{:.4f}",
        "95% CI Lower":"{:.4f}","95% CI Upper":"{:.4f}","Margin of Error":"{:.5f}"}),
        use_container_width=True)

    fig = go.Figure()
    for _,row in ci_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["95% CI Lower"],row["Mean"],row["95% CI Upper"]],
            y=[row["Variable"]]*3, mode="lines+markers",
            line=dict(color=PURPLE, width=2),
            marker=dict(size=[8,14,8], color=[PURPLE,"#ffffff",PURPLE]),
            name=row["Variable"], showlegend=False))
    fig.update_layout(**DARK, title="95% Confidence Intervals", xaxis_title="Value", height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-hdr">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = d[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0,"#0d0d0d"],[0.5,"#2a0000"],[1,"#e50914"]],
        text=corr.values.round(3), texttemplate="%{text}",
        textfont_size=12, showscale=True, zmin=-1, zmax=1))
    fig.update_layout(**DARK, title="Pearson Correlation Matrix", height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight"><b>Key Insight:</b> Patients and Vehicles Involved show positive correlation. Fatal accidents have distinctly higher response times and patient counts.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# GROWTH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Growth Analysis":
    st.markdown('<div class="page-title">Growth Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Year-over-Year and Monthly Trends</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    yr_df = d.groupby("Year").agg(
        Accidents=("Year","count"), Fatal=("Is_Fatal","sum"),
        Avg_Patients=("Patients","mean"), Avg_Response=("responsetime","mean")
    ).reset_index()
    yr_df["YoY_%"] = yr_df["Accidents"].pct_change()*100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65,0.35],
                        subplot_titles=("Accidents and Fatal Cases per Year","YoY Change (%)"))
    fig.add_trace(go.Bar(x=yr_df["Year"], y=yr_df["Accidents"], name="Total Accidents",
                         marker_color=PURPLE, opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=yr_df["Year"], y=yr_df["Fatal"], name="Fatal Cases",
                             mode="lines+markers", line=dict(color="#ffa94d", width=2.5),
                             marker=dict(size=9)), row=1, col=1)
    c_yoy = [PURPLE if x < 0 else "#69db7c" for x in yr_df["YoY_%"].fillna(0)]
    fig.add_trace(go.Bar(x=yr_df["Year"], y=yr_df["YoY_%"], name="YoY %",
                         marker_color=c_yoy), row=2, col=1)
    fig.update_layout(**DARK, height=500)
    st.plotly_chart(fig, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        mon_df = d.groupby("Month").agg(
            Accidents=("Year","count"), Fatal=("Is_Fatal","sum")).reset_index()
        mon_df["Month_Name"] = mon_df["Month"].map({1:"Jan",2:"Feb",3:"Mar",4:"Apr",
            5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mon_df["Month_Name"], y=mon_df["Accidents"],
                                 fill="tozeroy", mode="lines+markers",
                                 line=dict(color=PURPLE, width=2.5),
                                 fillcolor="rgba(191,64,255,0.12)", name="Accidents"))
        fig.update_layout(**DARK, title="Monthly Accident Pattern", height=340)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        season_yr = d.groupby(["Year","Season"]).size().reset_index(name="Accidents")
        fig = px.line(season_yr, x="Year", y="Accidents", color="Season",
                      color_discrete_sequence=COLORS, title="Season-wise Yearly Trend", markers=True)
        fig.update_layout(**DARK, height=340)
        st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yr_df["Year"], y=yr_df["Avg_Response"],
                             mode="lines+markers+text",
                             text=yr_df["Avg_Response"].round(1),
                             textposition="top center",
                             textfont=dict(color="#ffffff", size=10),
                             line=dict(color="#4dabf7", width=2.5),
                             marker=dict(size=9, color=PURPLE),
                             fill="tozeroy", fillcolor="rgba(191,64,255,0.08)"))
    fig.update_layout(**DARK, title="Average Emergency Response Time — Year-over-Year",
                      yaxis_title="Avg Response Time (min)", height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight"><b>Trend Insight:</b> 2022 recorded the highest accident count. Spring and Summer show elevated frequencies. Response time improvements directly impact patient survival rates.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PROBABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Probability Analysis":
    st.markdown('<div class="page-title">Probability Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Poisson, Normal, Chi-Square and Bayes Theorem</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-hdr">Poisson Distribution — Patients per Accident</div>', unsafe_allow_html=True)
        lam = d["Patients"].mean()
        obs = d["Patients"].value_counts().sort_index()
        k_range = np.arange(0, min(15, obs.index.max()+2))
        pmf = stats.poisson.pmf(k_range, mu=lam)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=obs.index, y=obs.values/len(d), name="Observed Freq",
                             marker_color=PURPLE, opacity=0.65))
        fig.add_trace(go.Scatter(x=k_range, y=pmf, mode="lines+markers",
                                 name=f"Poisson (lambda={lam:.2f})",
                                 line=dict(color="#ffffff", width=2),
                                 marker=dict(size=6, color="#ffffff")))
        fig.update_layout(**DARK, title=f"Poisson Fit — lambda = {lam:.3f}",
                          xaxis_title="Patients per Accident", yaxis_title="Probability", height=360)
        st.plotly_chart(fig, use_container_width=True)
        ca,cb,cc = st.columns(3)
        ca.metric("P(X = 1)", f"{stats.poisson.pmf(1, lam):.4f}")
        cb.metric("P(X <= 2)", f"{stats.poisson.cdf(2, lam):.4f}")
        cc.metric("P(X >= 3)", f"{1-stats.poisson.cdf(2, lam):.4f}")

    with c2:
        st.markdown('<div class="sec-hdr">Normal Distribution — Response Time</div>', unsafe_allow_html=True)
        resp = d["responsetime"].dropna()
        mu_r, sd_r = resp.mean(), resp.std()
        x_r = np.linspace(resp.min(), resp.max(), 300)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=resp, nbinsx=40, histnorm="probability density",
                                   name="Observed", marker_color=PURPLE, opacity=0.55))
        fig.add_trace(go.Scatter(x=x_r, y=stats.norm.pdf(x_r, mu_r, sd_r), mode="lines",
                                 name=f"N(mu={mu_r:.1f}, sigma={sd_r:.1f})",
                                 line=dict(color="#ffffff", width=2)))
        fig.update_layout(**DARK, title="Normal Fit — Emergency Response Time",
                          xaxis_title="Response Time (min)", yaxis_title="Density", height=360)
        st.plotly_chart(fig, use_container_width=True)
        sw_s, sw_p = stats.shapiro(resp.sample(min(200,len(resp)), random_state=42))
        ca,cb,cc = st.columns(3)
        ca.metric("Mean", f"{mu_r:.2f} min")
        cb.metric("Std Dev", f"{sd_r:.2f}")
        cc.metric("Shapiro p-val", f"{sw_p:.4f}")

    st.markdown('<div class="sec-hdr">Interactive Probability Calculator</div>', unsafe_allow_html=True)
    cc1,cc2,cc3 = st.columns(3)
    with cc1:
        dist_sel = st.radio("Distribution", ["Poisson (Patients)","Normal (Response Time)"])
    with cc2:
        if "Poisson" in dist_sel:
            k_val = st.number_input("Value of k", min_value=0, max_value=20, value=1)
        else:
            r_lo = st.number_input("Lower bound (min)", value=5.0, step=1.0)
    with cc3:
        if "Poisson" in dist_sel:
            k_cum = st.number_input("Cumulative k <=", min_value=0, max_value=20, value=2)
        else:
            r_hi = st.number_input("Upper bound (min)", value=20.0, step=1.0)

    if st.button("Calculate", use_container_width=True):
        if "Poisson" in dist_sel:
            r1,r2,r3 = st.columns(3)
            r1.metric(f"P(X = {k_val})", f"{stats.poisson.pmf(k_val, lam):.5f}")
            r2.metric(f"P(X <= {k_cum})", f"{stats.poisson.cdf(k_cum, lam):.5f}")
            r3.metric(f"P(X > {k_cum})", f"{1-stats.poisson.cdf(k_cum, lam):.5f}")
        else:
            p_rng = stats.norm.cdf(r_hi,mu_r,sd_r) - stats.norm.cdf(r_lo,mu_r,sd_r)
            r1,r2,r3 = st.columns(3)
            r1.metric(f"P({r_lo:.0f} to {r_hi:.0f})", f"{p_rng:.5f}")
            r2.metric(f"P(X > {r_hi:.0f})", f"{1-stats.norm.cdf(r_hi,mu_r,sd_r):.5f}")
            r3.metric(f"P(X < {r_lo:.0f})", f"{stats.norm.cdf(r_lo,mu_r,sd_r):.5f}")

    st.markdown('<div class="sec-hdr">Chi-Square Test — Weather vs Accident Class</div>', unsafe_allow_html=True)
    ct = pd.crosstab(d["Weather"], d["ACCLASS"])
    st.dataframe(ct, use_container_width=True)
    chi2_val, p_chi, dof, _ = stats.chi2_contingency(ct)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Chi-Square Statistic", f"{chi2_val:.4f}")
    c2.metric("p-value", f"{p_chi:.4f}")
    c3.metric("Degrees of Freedom", dof)
    c4.metric("Result (a=0.05)", "Significant" if p_chi<0.05 else "Not Significant")

    st.markdown('<div class="sec-hdr">Bayesian Analysis — P(Fatal | Cause)</div>', unsafe_allow_html=True)
    p_fatal = d["Is_Fatal"].mean()
    bayes_rows = []
    for cause in d["Cause"].unique():
        p_c = (d["Cause"]==cause).mean()
        if p_c > 0:
            p_cf = ((d["Is_Fatal"]==1) & (d["Cause"]==cause)).mean()
            p_fc = p_cf/p_c if p_c>0 else 0
            bayes_rows.append({"Cause":cause,"P(Cause)":round(p_c,4),
                                "P(Fatal)":round(p_fatal,4),"P(Fatal|Cause)":round(p_fc,4),
                                "Risk Lift":round(p_fc/p_fatal,3) if p_fatal>0 else 0})
    bayes_df = pd.DataFrame(bayes_rows).sort_values("P(Fatal|Cause)", ascending=False)
    st.dataframe(bayes_df.style.format({"P(Cause)":"{:.4f}","P(Fatal)":"{:.4f}",
        "P(Fatal|Cause)":"{:.4f}","Risk Lift":"{:.3f}"}),
        use_container_width=True)
    fig = px.bar(bayes_df, x="Cause", y="P(Fatal|Cause)", color="Risk Lift",
                 color_continuous_scale=[[0,"#0d0d0d"],[1,"#e50914"]],
                 title="P(Fatal | Cause) — Bayesian Risk by Accident Cause")
    fig.add_hline(y=p_fatal, line_dash="dash", line_color="#444444",
                  annotation_text=f"Unconditional P(Fatal)={p_fatal:.4f}",
                  annotation_font_color="#aaaaaa")
    fig.update_layout(**DARK, height=360, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction Model":
    st.markdown('<div class="page-title">Prediction Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Linear, Polynomial and Multiple Regression</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-hdr">Linear Regression — Response Time vs Patients</div>', unsafe_allow_html=True)
        sub = d[["responsetime","Patients"]].dropna()
        X = sub[["responsetime"]].values; y = sub["Patients"].values
        lr = LinearRegression().fit(X, y)
        y_pred = lr.predict(X)
        r2   = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        x_line = np.linspace(sub["responsetime"].min(), sub["responsetime"].max(), 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub["responsetime"], y=sub["Patients"], mode="markers",
                                 marker=dict(color=PURPLE, opacity=0.2, size=4), name="Data"))
        fig.add_trace(go.Scatter(x=x_line, y=lr.predict(x_line.reshape(-1,1)), mode="lines",
                                 line=dict(color="#ffffff", width=2.5),
                                 name=f"y={lr.intercept_:.3f}+{lr.coef_[0]:.4f}x"))
        fig.update_layout(**DARK, title="Response Time vs Patients",
                          xaxis_title="Response Time (min)", yaxis_title="Patients", height=360)
        st.plotly_chart(fig, use_container_width=True)
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("Intercept (B0)", f"{lr.intercept_:.4f}")
        mc2.metric("Slope (B1)", f"{lr.coef_[0]:.4f}")
        mc3.metric("R Squared", f"{r2:.4f}")
        mc4.metric("RMSE", f"{rmse:.4f}")

    with c2:
        st.markdown('<div class="sec-hdr">Polynomial Regression — Vehicles vs Patients</div>', unsafe_allow_html=True)
        X2 = d[["Vehicles_Involved"]].values; y2 = d["Patients"].values
        poly = PolynomialFeatures(degree=2)
        X2p  = poly.fit_transform(X2)
        lr2  = LinearRegression().fit(X2p, y2)
        r2p  = r2_score(y2, lr2.predict(X2p))
        x_p  = np.linspace(1, d["Vehicles_Involved"].max(), 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["Vehicles_Involved"], y=d["Patients"], mode="markers",
                                 marker=dict(color=PURPLE, opacity=0.2, size=4), name="Data"))
        fig.add_trace(go.Scatter(x=x_p, y=lr2.predict(poly.transform(x_p.reshape(-1,1))),
                                 mode="lines", line=dict(color="#ffa94d", width=2.5),
                                 name=f"Poly R2={r2p:.4f}"))
        fig.update_layout(**DARK, title="Vehicles Involved vs Patients (Poly deg=2)",
                          xaxis_title="Vehicles Involved", yaxis_title="Patients", height=360)
        st.plotly_chart(fig, use_container_width=True)
        mc1,mc2 = st.columns(2)
        mc1.metric("Polynomial R Squared", f"{r2p:.4f}")
        mc2.metric("RMSE", f"{np.sqrt(mean_squared_error(y2, lr2.predict(X2p))):.4f}")

    st.markdown('<div class="sec-hdr">Residual Analysis</div>', unsafe_allow_html=True)
    residuals = y - y_pred
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Residuals vs Fitted","Residual Distribution"))
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                             marker=dict(color=PURPLE, opacity=0.3, size=4)), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#444444", row=1, col=1)
    fig.add_trace(go.Histogram(x=residuals, nbinsx=30, marker_color=PURPLE, opacity=0.7), row=1, col=2)
    fig.update_layout(**DARK, height=320, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-hdr">Live Patient Count Predictor</div>', unsafe_allow_html=True)
    p1,p2 = st.columns(2)
    with p1:
        pred_resp = st.slider("Response Time (min)", 1, 60, 15)
    with p2:
        pred_veh  = st.slider("Vehicles Involved", 1, 7, 2)

    Xm  = d[["responsetime","Vehicles_Involved"]].dropna().values
    ym  = d.loc[d[["responsetime","Vehicles_Involved"]].dropna().index,"Patients"].values
    lrm = LinearRegression().fit(Xm, ym)
    pred_val = max(1, lrm.predict([[pred_resp, pred_veh]])[0])
    r2m = r2_score(ym, lrm.predict(Xm))
    rr1,rr2,rr3 = st.columns(3)
    rr1.metric("Predicted Patients", f"{pred_val:.2f}")
    rr2.metric("Multiple R Squared", f"{r2m:.4f}")
    rr3.metric("Severity Risk", "High" if pred_val>=3 else "Medium" if pred_val>=2 else "Low")
    st.markdown(f'<div class="insight">With response time of <b>{pred_resp} min</b> and <b>{pred_veh}</b> vehicles involved, the model predicts approximately <b>{pred_val:.1f} patients</b> in emergency.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BEST TIME TO TRAVEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Best Time to Travel":
    st.markdown('<div class="page-title">Best Time to Travel</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Time-based Accident Risk Patterns</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Accident Risk by Time Slot and Day</div>', unsafe_allow_html=True)
    pivot_td = d.pivot_table(index="Day", columns="Time", values="Is_Fatal",
                              aggfunc="count", fill_value=0)
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot_td = pivot_td.reindex([x for x in day_order if x in pivot_td.index])
    fig = go.Figure(go.Heatmap(
        z=pivot_td.values, x=list(pivot_td.columns), y=list(pivot_td.index),
        colorscale=[[0,"#0d0d0d"],[0.3,"#2a0000"],[0.7,"#990000"],[1,"#e50914"]],
        text=pivot_td.values, texttemplate="%{text}", textfont_size=9,
        showscale=True, colorbar=dict(title="Accidents", tickfont=dict(color="#aaaaaa"))))
    fig.update_layout(**DARK, title="Accidents Heatmap — Time Slot x Day of Week",
                      xaxis_title="Time Slot", yaxis_title="Day of Week", height=380)
    st.plotly_chart(fig, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        time_df = d.groupby("Time").agg(
            Accidents=("Year","count"), Fatal=("Is_Fatal","sum")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=time_df["Time"], y=time_df["Accidents"],
                             marker_color=PURPLE, opacity=0.8, name="Accidents"))
        fig.add_trace(go.Scatter(x=time_df["Time"], y=time_df["Fatal"],
                                 mode="lines+markers", line=dict(color="#ffffff", width=2),
                                 marker=dict(size=7), name="Fatal"))
        fig.update_layout(**DARK, title="Accidents and Fatal Cases by Time Slot", height=340)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        day_df = d.groupby("Day").agg(
            Accidents=("Year","count"), Fatal=("Is_Fatal","sum")).reindex(day_order).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=day_df["Day"], y=day_df["Accidents"],
                             marker_color=PURPLE, opacity=0.8, name="Accidents"))
        fig.update_layout(**DARK, title="Accidents by Day of Week", height=340)
        st.plotly_chart(fig, use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        lgt_df = d.groupby("Light").agg(
            Accidents=("Year","count"), Fatal=("Is_Fatal","sum")).reset_index()
        fig = px.bar(lgt_df, x="Light", y="Accidents", color="Fatal",
                     color_continuous_scale=[[0,"#0d0d0d"],[1,"#e50914"]],
                     title="Accidents by Light Condition")
        fig.update_layout(**DARK, height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        sea_df = d.groupby("Season").agg(
            Accidents=("Year","count"), Fatal=("Is_Fatal","sum")).reset_index()
        fig = px.bar(sea_df, x="Season", y="Accidents", color="Fatal",
                     color_continuous_scale=[[0,"#0d0d0d"],[1,"#e50914"]],
                     title="Accidents by Season")
        fig.update_layout(**DARK, height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight"><b>Safety Insight:</b> The 12:00 to 18:00 slot records the highest accident count. Night-time accidents (18:00 to 24:00) have a disproportionately higher fatal rate. Friday sees the most accidents, especially in the evening.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEVERITY EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Severity Explorer":
    st.markdown('<div class="page-title">Severity Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Deep-Dive into Accident Severity Patterns</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    inj_order  = ["Minor","Single Fracture","Multiple Fractures","Head Injury","Spinal Injury"]
    inj_colors = {"Minor":"#69db7c","Single Fracture":"#ffe066",
                  "Multiple Fractures":"#ffa94d","Head Injury":"#ff6b6b","Spinal Injury":"#e50914"}

    c1,c2 = st.columns(2)
    with c1:
        inj_df = d["InjuryType"].value_counts().reindex(inj_order, fill_value=0).reset_index()
        inj_df.columns = ["InjuryType","Count"]
        fig = px.bar(inj_df, x="Count", y="InjuryType", orientation="h",
                     color="InjuryType", color_discrete_map=inj_colors,
                     title="Injury Type Distribution")
        fig.update_layout(**DARK, height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        inj_sea = d.groupby(["Season","InjuryType"]).size().reset_index(name="Count")
        fig = px.bar(inj_sea, x="Season", y="Count", color="InjuryType",
                     color_discrete_map=inj_colors, barmode="stack",
                     title="Injury Type Stack by Season")
        fig.update_layout(**DARK, height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-hdr">Response Time Distribution by Injury Type</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for inj in inj_order:
        vals = d[d["InjuryType"]==inj]["responsetime"].dropna()
        if len(vals) > 10:
            fig.add_trace(go.Violin(x=vals, name=inj,
                                    line_color=inj_colors.get(inj, PURPLE),
                                    fillcolor=inj_colors.get(inj, PURPLE),
                                    opacity=0.6, meanline_visible=True))
    fig.update_layout(**DARK, title="Response Time per Injury Type",
                      xaxis_title="Response Time (min)", height=360)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-hdr">Cause vs Injury Type Heatmap</div>', unsafe_allow_html=True)
    pivot_ci = d.pivot_table(index="Cause", columns="InjuryType",
                              values="Is_Fatal", aggfunc="count", fill_value=0)
    pivot_ci = pivot_ci.reindex(columns=[c for c in inj_order if c in pivot_ci.columns])
    fig = go.Figure(go.Heatmap(
        z=pivot_ci.values, x=list(pivot_ci.columns), y=list(pivot_ci.index),
        colorscale=[[0,"#0d0d0d"],[1,"#e50914"]],
        text=pivot_ci.values, texttemplate="%{text}", textfont_size=11))
    fig.update_layout(**{**DARK, "margin": dict(t=50,b=60,l=160,r=20)},
                      title="Accident Count — Cause vs Injury Type", height=360)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-hdr">Severity Summary by Injury Type</div>', unsafe_allow_html=True)
    sev_stats = d.groupby("InjuryType").agg(
        Count=("InjuryType","count"), Fatal_Cases=("Is_Fatal","sum"),
        Avg_Patients=("Patients","mean"), Avg_Response=("responsetime","mean"),
        Avg_Vehicles=("Vehicles_Involved","mean")
    ).reindex([i for i in inj_order if i in d["InjuryType"].unique()]).round(3)
    sev_stats["Fatal_%"] = (sev_stats["Fatal_Cases"]/sev_stats["Count"]*100).round(2)
    st.dataframe(sev_stats.style.format("{:.3f}"), use_container_width=True)
    st.markdown('<div class="insight"><b>Key Finding:</b> Head Injury and Spinal Injury cases have significantly higher response times and patient counts. Over Speed is the dominant cause across all injury types, accounting for the majority of severe cases.</div>', unsafe_allow_html=True)
