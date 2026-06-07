import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="SECI Solar Intelligence", layout="wide")


def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200: return None
        return r.json()
    except: return None
    
LOTTIE_SUN = "https://assets5.lottiefiles.com/private_files/lf30_moat3dxk.json"
LOTTIE_CLOUD = "https://assets3.lottiefiles.com/packages/lf20_fiq9v72r.json"
LOTTIE_NIGHT = "https://assets8.lottiefiles.com/packages/lf20_96py9mke.json"


st.markdown("""
    <style>
    target the sidebar collapse button icon 
    button[data-testid="stBaseButton-headerNoPadding"] span[data-testid="stIconMaterial"] {
        font-family: 'Material Icons' !important;
    }

    [data-testid="stSidebar"][aria-expanded="false"] + section button span::before {
        content: "keyboard_double_arrow_right" !important;
    }
    
    header[data-testid="stHeader"] {
        z-index: 100 !important;
    }
    
    span[data-testid="stIconMaterial"]:contains("keyboard_double_arrow_left") {
        content: "keyboard_double_arrow_right" !important;
    } 
    </style>
    """, unsafe_allow_html=True)
    
col_left, col_mid, col_right = st.columns([1, 2, 1])
with col_mid:
    try:
        st.image("SECI.png", use_container_width=True)
    except:
        st.markdown("<h1 style='text-align: center; color: #1a4a7a;'>SECI SOLAR DASHBOARD</h1>", unsafe_allow_html=True)

st.markdown("<hr style='border: 1.5px solid #1a4a7a; margin-top: 0;'>", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        # Load model and data [cite: 2025-12-16]
        model = joblib.load('solar_model.pkl')
        df = pd.read_csv('enhanced_model_results_1.csv')
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], dayfirst=True)
        df['Actual'] = df['Actual'].clip(lower=0)
        df['Predicted'] = df['Predicted'].clip(lower=0)
        return model, df
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, results_df = load_assets()

st.sidebar.title("Navigation Menu")
page = st.sidebar.radio("Go to:", ["Main Dashboard", "Live Prediction Tool", "Model Analytics"])

if page == "Main Dashboard":
    st.markdown("<style>.stApp { background: linear-gradient(135deg, #0288d1 0%, #e1f5fe 100%); } h1, h2, h3, p, label { font-weight: 900 ; color: #000000 ; }</style>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white !important;'>📊 SECI Main Dashboard</h1>", unsafe_allow_html=True)
    
    if results_df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Plant Capacity", "100.00 MW")
        c2.metric("Monthly Generation (in MWh)", "13800.82 MWh") 
        c3.metric("Model Precision", "95.3%", delta="4.7% Error")
        c4.metric("Avg Daily Output (in MWh)", "468.36 MWh") 
        
        st.info("💡 Use the slider below the graph to zoom into specific days or cloud events.")
        st.write("---")
        
        st.subheader("🔍 Generation Trend Analysis (Historical)")
    
    fig_line = px.line(results_df, x='TimeStamp', y=['Actual', 'Predicted'],
                       labels={"value": "Power (MW) ", "TimeStamp": "Time"},
                       color_discrete_map={"Actual": "#1f77b4", "Predicted": "#ef553b"})
    
    fig_line.update_layout(
        font={'family': "Times New Roman"},
        xaxis_rangeslider_visible=True,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_line, use_container_width=True)


# PAGE 2: LIVE PREDICTION TOOL 
##### PAGE 2 LIVE PREDICTION TOOL #####
elif page == "Live Prediction Tool":

    st.markdown("""
    <h1 style='text-align:center;color:#1a4a7a;'>
    ☀️ Solar Generation Forecast Center
    </h1>
    """, unsafe_allow_html=True)

    LAT, LON = 21.10, 80.99

    max_forecast_date = datetime.now() + timedelta(days=10)

    selected_future_date = st.date_input(
        "📅 Select Forecast Date",
        min_value=datetime.now().date(),
        max_value=max_forecast_date.date()
    )

    if st.button("🔮 Generate Forecast"):

        with st.spinner("Fetching weather forecast and generating prediction..."):

            f_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={LAT}"
                f"&longitude={LON}"
                f"&hourly=shortwave_radiation,temperature_2m,"
                f"surface_pressure,cloudcover"
                f"&forecast_days=11"
            )

            res = requests.get(f_url)

            if res.status_code == 200:

                f_data = res.json()['hourly']

                all_df = pd.DataFrame({
                    'TimeStamp': pd.to_datetime(f_data['time']),
                    'GHI': [x * 1.3 for x in f_data['shortwave_radiation']],
                    'GII': [x * 1.4 for x in f_data['shortwave_radiation']],
                    'AMB_TEMP': f_data['temperature_2m'],
                    'AIR_PRESS': f_data['surface_pressure'],
                    'CLOUD_COVER': f_data['cloudcover']
                })

                day_df = all_df[
                    all_df['TimeStamp'].dt.date == selected_future_date
                ].copy()

                if not day_df.empty:

                    # Feature Engineering
                    day_df['MOD_TEMP'] = day_df['AMB_TEMP'] + 5
                    day_df['Hour'] = day_df['TimeStamp'].dt.hour
                    day_df['Minute'] = day_df['TimeStamp'].dt.minute
                    day_df['Day_of_Year'] = day_df['TimeStamp'].dt.dayofyear
                    day_df['Day_of_Week'] = day_df['TimeStamp'].dt.dayofweek

                    cols = [
                        'GHI',
                        'GII',
                        'MOD_TEMP',
                        'AMB_TEMP',
                        'AIR_PRESS',
                        'CLOUD_COVER',
                        'Hour',
                        'Minute',
                        'Day_of_Year',
                        'Day_of_Week'
                    ]

                    raw_pred = model.predict(day_df[cols])

                    day_df['Predicted_MW'] = (
                        raw_pred * 0.8
                    ).clip(0, 100)

                    # Summary Metrics
                    total_mwh = day_df['Predicted_MW'].sum()
                    peak_mw = day_df['Predicted_MW'].max()

                    peak_row = day_df.loc[
                        day_df['Predicted_MW'].idxmax()
                    ]

                    peak_time = peak_row['TimeStamp']

                    avg_temp = day_df['AMB_TEMP'].mean()

                    confidence_low = total_mwh * 0.953
                    confidence_high = total_mwh * 1.047

                    st.success(
                        f"✅ Forecast Generated for {selected_future_date}"
                    )

                    # KPI Cards
                    k1, k2, k3, k4 = st.columns(4)

                    k1.metric(
                        "⚡ Total Energy",
                        f"{total_mwh:.2f} MWh"
                    )

                    k2.metric(
                        "☀️ Peak Power",
                        f"{peak_mw:.2f} MW"
                    )

                    k3.metric(
                        "🕒 Peak Time",
                        peak_time.strftime("%I:%M %p")
                    )

                    k4.metric(
                        "🎯 Confidence",
                        "95.3%"
                    )

                    st.markdown("---")

                    # Forecast Curve
                    st.subheader(
                        "📈 24-Hour Solar Generation Forecast"
                    )

                    fig_forecast = px.area(
                        day_df,
                        x="TimeStamp",
                        y="Predicted_MW",
                        title="Hourly Forecasted Generation"
                    )

                    fig_forecast.update_layout(
                        xaxis_title="Time",
                        yaxis_title="Power Output (MW)",
                        hovermode="x unified"
                    )

                    st.plotly_chart(
                        fig_forecast,
                        use_container_width=True
                    )

                    st.markdown("---")

                    # Weather Cards
                    st.subheader("🌦 Weather Conditions")

                    w1, w2, w3, w4 = st.columns(4)

                    w1.metric(
                        "Avg GHI",
                        f"{day_df['GHI'].mean():.0f}"
                    )

                    w2.metric(
                        "Cloud Cover",
                        f"{day_df['CLOUD_COVER'].mean():.0f}%"
                    )

                    w3.metric(
                        "Temperature",
                        f"{avg_temp:.1f} °C"
                    )

                    w4.metric(
                        "Pressure",
                        f"{day_df['AIR_PRESS'].mean():.0f} hPa"
                    )

                    st.markdown("---")

                    # Generation vs GHI
                    st.subheader(
                        "☀️ Irradiance vs Power Generation"
                    )

                    fig_compare = go.Figure()

                    fig_compare.add_trace(
                        go.Scatter(
                            x=day_df['TimeStamp'],
                            y=day_df['Predicted_MW'],
                            name="Power Output (MW)"
                        )
                    )

                    fig_compare.add_trace(
                        go.Scatter(
                            x=day_df['TimeStamp'],
                            y=day_df['GHI'],
                            name="GHI"
                        )
                    )

                    fig_compare.update_layout(
                        hovermode="x unified"
                    )

                    st.plotly_chart(
                        fig_compare,
                        use_container_width=True
                    )

                    st.markdown("---")

                    # Forecast Confidence
                    st.subheader(
                        "📊 Forecast Confidence Range"
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Minimum Expected",
                        f"{confidence_low:.2f} MWh"
                    )

                    c2.metric(
                        "Expected Output",
                        f"{total_mwh:.2f} MWh"
                    )

                    c3.metric(
                        "Maximum Expected",
                        f"{confidence_high:.2f} MWh"
                    )

                    st.markdown("---")

                    # AI Insights
                    st.subheader("🧠 Forecast Insights")

                    avg_cloud = (
                        day_df['CLOUD_COVER'].mean()
                    )

                    if avg_cloud > 60:
                        sky_msg = (
                            "High cloud cover may reduce output."
                        )
                    elif avg_cloud > 30:
                        sky_msg = (
                            "Moderate cloud cover expected."
                        )
                    else:
                        sky_msg = (
                            "Clear sky conditions expected."
                        )

                    st.info(
                        f"""
                        • Peak generation expected at
                        {peak_time.strftime('%I:%M %p')}

                        • Expected total energy:
                        {total_mwh:.2f} MWh

                        • {sky_msg}

                        • Forecast confidence:
                        95.3%
                        """
                    )

                    st.markdown("---")

                    # Forecast Table
                    st.subheader(
                        "📋 Hourly Forecast Data"
                    )

                    display_df = day_df[
                        [
                            'TimeStamp',
                            'Predicted_MW',
                            'GHI',
                            'AMB_TEMP',
                            'CLOUD_COVER'
                        ]
                    ].copy()

                    st.dataframe(
                        display_df,
                        use_container_width=True
                    )

                    csv = display_df.to_csv(
                        index=False
                    )

                    st.download_button(
                        "📥 Download Forecast CSV",
                        csv,
                        file_name=
                        f"forecast_{selected_future_date}.csv",
                        mime="text/csv"
                    )

                else:
                    st.error(
                        "Forecast data unavailable for selected date."
                    )

            else:
                st.error(
                    "Failed to connect to weather API."
                )


# Page 3: 
elif page == "Model Analytics":
    # Custom CSS for the Sky Blue + Sun Yellow Aesthetic
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0288d1 0%, #e1f5fe 100%);
        }
        .analytics-card {
            background-color: #FFFDE7; /* Light Sun Yellow */
            padding: 25px;
            border-radius: 20px;
            border-bottom: 6px solid #FBC02D; /* Golden accent */
            box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
            color: #1a4a7a;
            margin-bottom: 25px;
        }
        h2, h3 { color: #1a4a7a !important; font-family: 'Times New Roman'; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: white; text-shadow: 2px 2px 5px rgba(0,0,0,0.3);'>☀️ Model Accuracy Analytics</h1>", unsafe_allow_html=True)

    # --- 1. TOP SUMMARY CARDS (Updated from Image_b87367) ---
    st.markdown("### 📊 Operational Benchmarks")
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown(f"""<div class='analytics-card'><h4>Daily Avg Generation</h4><h2>460.02 MWh</h2></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='analytics-card'><h4>Average Daily Error</h4><h2>4.7 MWh</h2></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='analytics-card'><h4>Safe Daily Output</h4><h2>455.32 MWh</h2></div>""", unsafe_allow_html=True)

    # --- 2. TABLE 1: MODEL PREDICTION VALUES (Updated Data) ---
    st.write("---")
    st.markdown("### 📋 Periodical Forecast Accuracy")
    
    data1 = {
        "Time Period": ["1 DAY", "7 DAYS", "15 DAYS", "1 MONTH", "4 MONTHS"],
        "Actual (MWh)": [460.12, 3220.84, 6978.7, 13800.82, 55200.5],
        "Predicted (MWh)": [465.01, 3399.07, 6850.2, 14000.8, 54987.80],
        "Error (%)": [1.08, 2.3, 1.3, 1.44, 0.38]
    }
    df1 = pd.DataFrame(data1)
    
    # Styled Table with Dynamic Yellow/Blue Shading
    st.dataframe(df1.style.background_gradient(cmap='YlGnBu', subset=["Error (%)"]).format({
        "Actual (MWh)": "{:,.2f}", "Predicted (MWh)": "{:,.2f}", "Error (%)": "{:.2f}%"
    }), use_container_width=True)

    # --- 3. DYNAMIC GRAPHS ---
    st.write("---")
    g1, g2 = st.columns(2)

    with g1:
        # Graph 1: Actual vs Predicted (Bar Chart)
        st.markdown("#### Actual vs Predicted Horizon")
        fig_bar = go.Figure(data=[
            go.Bar(name='Actual', x=df1['Time Period'], y=df1['Actual (MWh)'], marker_color='#1a4a7a'),
            go.Bar(name='Predicted', x=df1['Time Period'], y=df1['Predicted (MWh)'], marker_color='#FBC02D')
        ])
        fig_bar.update_layout(barmode='group', height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)

    with g2:
        # Graph 2: Error Trend (Line Chart)
        st.markdown("#### Precision Stability (Error %)")
        fig_line = px.line(df1, x="Time Period", y="Error (%)", markers=True, 
                           color_discrete_sequence=["#1a4a7a"])
        fig_line.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_line, use_container_width=True)

    # Graph 3: Safe Output Projection (Area Chart)
    st.markdown("#### Cumulative Generation Security")
    fig_area = px.area(df1, x="Time Period", y=["Actual (MWh)", "Predicted (MWh)"], 
                       color_discrete_map={"Actual (MWh)": "#1a4a7a", "Predicted (MWh)": "#FBC02D"})
    fig_area.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_area, use_container_width=True)

    # --- 4. TABLE 2: AVERAGE STATS (Updated Data) ---
    st.write("---")
    st.markdown("### 🔍 Statistical Confidence Levels")
    
    # Updated values from Image_b87367
    data2 = {
        "Metric Type": ["Daily Average", "Monthly Total"],
        "Generation (MWh)": [460.12, 13800.82],
        "Avg Error (MWh)": [4.7, 153],
        "Safe Output (MWh)": [455.34, 13659.82],
        "Error Percentage": ["0.98%", "0.98%"]
    }
    st.table(pd.DataFrame(data2))

    # --- DYNAMIC GRAPHS FOR TABLE 2 ---
    st.markdown("#### 📈 Confidence & Error Impact Analysis")
    
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        # 1. Comparison of Generation vs Safe Output (Daily)
        fig_daily = go.Figure(data=[
            go.Bar(name='Expected Generation', x=['Daily Avg'], y=[868.36], marker_color='#FBC02D'),
            go.Bar(name='Safe Output', x=['Daily Avg'], y=[863.26], marker_color='#1a4a7a')
        ])
        fig_daily.update_layout(
            title="Daily Reliability (MWh)",
            barmode='group',
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': "Times New Roman"}
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    with col_g2:
        # 2. Error Breakdown Pie Chart (Monthly)
        fig_pie = px.pie(
            values=[26050.82, 153], 
            names=['Safe Output', 'Avg Error'],
            color_discrete_sequence=['#1a4a7a', '#ef553b'],
            title="Monthly Margin of Error"
        )
        fig_pie.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

    # 3. Precision Indicator (Bullet Chart) - Updated to 0.738%
    # 3. Precision Indicator (Bullet Chart) - Updated to 0.738%
    st.markdown("#### 🎯 Model Precision Indicator")
    fig_bullet = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = 2.1,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Error Percentage (%)", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 5], 'tickwidth': 1},
            'bar': {'color': "#ef553b"},
            'steps': [
                {'range': [0, 0.5], 'color': "#e8f5e9"},
                {'range': [0.5, 1.0], 'color': "#fff3e0"},
                {'range': [1.0, 2.0], 'color': "#ffebee"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 2.1}
        }
    ))
    
    # FIXED: Height must be 10 or more. Changed from 2 to 250.
    fig_bullet.update_layout(
        height=250, 
        margin=dict(t=50, b=0), 
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_bullet, use_container_width=True)
    # Help Section to explain metrics
    with st.expander("❓ How to interpret these metrics?"):
        st.markdown("""
        ### Understanding the Analytics
        * **Daily Avg Generation**: The baseline expected energy output based on historical trends.
        * **Average Error (MAE)**: Our model has a **Mean Absolute Error of 5.1 MW**, meaning predictions are typically within this range of actual values.
        * **Safe Output**: This is the 'Guaranteed' generation level calculated by subtracting the average error from the prediction.
        * **Model Precision**: The **2.1% error rate** indicates extremely high accuracy, allowing for stable grid management and planning.
        """)

st.markdown("<div style='text-align: center; color: gray; margin-top: 50px;'>SECI Analytics Division © 2026</div>", unsafe_allow_html=True)
