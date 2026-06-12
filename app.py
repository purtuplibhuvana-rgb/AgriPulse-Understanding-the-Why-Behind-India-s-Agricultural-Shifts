import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as objects
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Crop Anomaly Attribution Engine",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
/* Font and base background */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body {margin:0; padding:0; font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #0f1319, #1a202c);
color: #e2e8f0;}
/* Card styling */
.anomaly-card, .top-finding-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(8px);
    border-radius: 12px;
    padding: 20px;
    margin-top: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    transition: transform 0.2s;
}
.anomaly-card:hover, .top-finding-card:hover {transform: translateY(-4px);}
.metric-container {display:flex; justify-content:space-between; margin-bottom:8px;}
.metric-label {color:#94a3b8; font-weight:500;}
.metric-value {color:#cbd5e1; font-weight:600;}
.highlight-green {color:#4ade80;}
.highlight-red {color:#f87171;}
.top-finding-card div {color:#cbd5e1;}
</style>
""", unsafe_allow_html=True)

# Helper function to load datasets
@st.cache_data
def load_data(workspace_dir):
    yield_csv = os.path.join(workspace_dir, "crop_yield.csv")
    report_json = os.path.join(workspace_dir, "anomaly_report.json")
    
    df_yield = pd.read_csv(yield_csv) if os.path.exists(yield_csv) else None
    
    if os.path.exists(report_json):
        with open(report_json, 'r') as f:
            report_data = json.load(f)
    else:
        report_data = []
        
    return df_yield, report_data

# Set workspace directory
WORKSPACE_DIR = os.getcwd()
df_yield, report_data = load_data(WORKSPACE_DIR)

if df_yield is None or len(report_data) == 0:
    st.error("Missing yield dataset or anomaly report! Please run Stage 1-5 pipeline first.")
    st.stop()

# Header Section
st.title("🌾 Crop Yield Anomaly Attribution Engine")
st.markdown("Automated Structural Break Detection & AI-Driven Attribution for Indian Agricultural Data")
st.markdown("---")

# Navigation Tabs
tab_explore, tab_top_findings = st.tabs(["🔍 Interactive Explorer", "🏆 Top 5 National Findings"])

# ----------------- TAB 1: INTERACTIVE EXPLORER -----------------
with tab_explore:
    col_side, col_main = st.columns([1, 3])
    
    with col_side:
        st.subheader("Select Parameters")
        
        # Get unique states and crops from report data (only those with anomalies)
        available_states = sorted(list(set(r['state'] for r in report_data)))
        selected_state = st.selectbox("State", available_states)
        
        available_crops = sorted(list(set(r['crop'] for r in report_data if r['state'] == selected_state)))
        selected_crop = st.selectbox("Crop", available_crops)
        
        # Get anomalies for selected state-crop
        anomalies = [r for r in report_data if r['state'] == selected_state and r['crop'] == selected_crop]
        
        if len(anomalies) > 0:
            st.success(f"Detected {len(anomalies)} structural break(s)!")
            # Select change point year
            cp_years = [r['break_year'] for r in anomalies]
            selected_year = st.selectbox("Select Change Point Year", cp_years)
            
            # Get specific anomaly card
            card = next(r for r in anomalies if r['break_year'] == selected_year)
        else:
            st.warning("No statistically significant breaks found for this combination.")
            card = None
            
    with col_main:
        # 1. Main Time Series Chart
        st.subheader(f"Yield Time Series — {selected_state} ({selected_crop})")
        
        # Get raw time series data
        ts_data = df_yield[(df_yield['State'] == selected_state) & (df_yield['Crop'] == selected_crop)].sort_values(['Crop_Year', 'Season']).copy()
        
        if len(ts_data) > 0:
            # Create a combined X label for readability
            ts_data['Year_Season'] = ts_data['Crop_Year'].astype(str) + " (" + ts_data['Season'] + ")"
            
            # Plotly Line Chart
            fig = objects.Figure()
            
            # Main Line
            fig.add_trace(objects.Scatter(
                x=ts_data['Year_Season'],
                y=ts_data['Yield'],
                mode='lines+markers',
                name='Yield',
                line=dict(color='#38bdf8', width=3),
                marker=dict(size=6, color='#0284c7'),
                hovertemplate="<b>Year/Season:</b> %{x}<br><b>Yield:</b> %{y:.2f} t/ha<extra></extra>"
            ))
            
            # Add vertical lines for change points
            for r in anomalies:
                # Find Year_Season corresponding to break_year and break_season
                break_ys = f"{r['break_year']} ({r['break_season']})"
                
                # Check if it exists in the data x values
                if break_ys in ts_data['Year_Season'].values:
                    fig.add_shape(
                        type="line",
                        x0=break_ys,
                        x1=break_ys,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(dash="dash", color="#f87171" if r['break_year'] == selected_year else "#94a3b8", width=2.5)
                    )
                    # Add a label for the break point
                    fig.add_annotation(
                        x=break_ys,
                        y=1.05,
                        xref="x",
                        yref="paper",
                        text=f"Break {r['break_year']}",
                        showarrow=False,
                        font=dict(color="#f87171" if r['break_year'] == selected_year else "#94a3b8")
                    )
            # End of break points loop
            # Update layout after adding all shapes/annotations
            fig.update_layout(
                plot_bgcolor='rgba(15, 19, 25, 0.8)',
                paper_bgcolor='rgba(15, 19, 25, 0.8)',
                margin=dict(l=40, r=40, t=20, b=40),
                height=400,
                font=dict(color='#e2e8f0'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickangle=-45),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Yield (tonnes/hectare)")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No raw time series data found.")

        # 2. Detail Section (Anomaly Card & SHAP Plot)
        if card:
            st.markdown("---")
            col_card, col_shap = st.columns([1.2, 1.8])
            
            with col_card:
                st.subheader("Anomaly Card")
                
                # Yield Change Highlight
                y_diff = card['yield_after'] - card['yield_before']
                y_pct = card['yield_pct_change']
                shift_class = "highlight-green" if y_diff > 0 else "highlight-red"
                shift_sign = "+" if y_diff > 0 else ""
                
                st.markdown(f"""
                <div class="anomaly-card">
                    <div class="card-title">Break Year: {card['break_year']} ({card['break_season']})</div>
                    <div class="metric-container">
                        <span class="metric-label">Yield Before (Mean)</span>
                        <span class="metric-value">{card['yield_before']:.3f} t/ha</span>
                    </div>
                    <div class="metric-container">
                        <span class="metric-label">Yield After (Mean)</span>
                        <span class="metric-value">{card['yield_after']:.3f} t/ha</span>
                    </div>
                    <div class="metric-container">
                        <span class="metric-label">Yield Shift</span>
                        <span class="{shift_class}">{shift_sign}{y_diff:.3f} t/ha ({shift_sign}{y_pct:.1f}%)</span>
                    </div>
                    <div class="metric-container">
                        <span class="metric-label">Welch t-test p-value</span>
                        <span class="metric-value">{card['p_value']:.4e}</span>
                    </div>
                    <div class="metric-container">
                        <span class="metric-label">Shift Magnitude (Std Dev)</span>
                        <span class="metric-value">{card['shift_in_std']:.2f} σ</span>
                    </div>
                    <div style="margin-top: 20px; font-style: italic; border-left: 3px solid #38bdf8; padding-left: 10px; color: #cbd5e1;">
                        "{card['explanation']}"
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # NDVI Context Layer Card
                st.subheader("NDVI Context Layer")
                ndvi_context = card['ndvi_context']
                if ndvi_context['has_context']:
                    st.info(ndvi_context['note'])
                    
                    # Label distribution Plotly Pie
                    dist_data = ndvi_context['label_distribution']
                    # Filter out zero categories
                    dist_data = {k: v for k, v in dist_data.items() if v > 0}
                    
                    fig_pie = px.pie(
                        names=list(dist_data.keys()),
                        values=list(dist_data.values()),
                        color_discrete_sequence=px.colors.sequential.Tealgrn,
                        height=220
                    )
                    fig_pie.update_layout(
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        legend=dict(font=dict(size=10, color='#e2e8f0'), orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.warning(ndvi_context['note'])
                    
            with col_shap:
                st.subheader("AI Attribution: SHAP Local Feature Shifts")
                
                # Features and SHAP diffs
                shap_details = card['shap_details']
                feat_names = shap_details['feature_names']
                shap_diffs = shap_details['shap_diffs']
                
                # Rename feature list for readability
                readable_names = {
                    'Area': 'Cultivation Area',
                    'Annual_Rainfall': 'Annual Rainfall',
                    'Fertilizer': 'Fertilizer Usage',
                    'Pesticide': 'Pesticide Application',
                    'Season_encoded': 'Seasonal Patterns',
                    'Crop_Year': 'Long-term Temporal Trend'
                }
                display_names = [readable_names.get(f, f) for f in feat_names]
                
                # Create DataFrame for plotting
                shap_df = pd.DataFrame({
                    'Feature': display_names,
                    'SHAP Shift': shap_diffs,
                    'Abs SHAP Shift': np.abs(shap_diffs)
                }).sort_values('Abs SHAP Shift', ascending=True)
                
                # Color code by shift direction
                shap_df['Color'] = shap_df['SHAP Shift'].apply(lambda x: '#4ade80' if x >= 0 else '#f87171')
                
                # Plotly Horizontal Bar Chart
                fig_shap = objects.Figure()
                fig_shap.add_trace(objects.Bar(
                    y=shap_df['Feature'],
                    x=shap_df['SHAP Shift'],
                    orientation='h',
                    marker_color=shap_df['Color'],
                    hovertemplate="<b>Feature:</b> %{y}<br><b>SHAP Impact Shift:</b> %{x:.4f}<extra></extra>"
                ))
                
                fig_shap.update_layout(
                    plot_bgcolor='rgba(15, 19, 25, 0.8)',
                    paper_bgcolor='rgba(15, 19, 25, 0.8)',
                    margin=dict(l=40, r=40, t=20, b=40),
                    height=350,
                    font=dict(color='#e2e8f0'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Attribution Change (After - Before)"),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                
                # Physical values comparison table
                st.markdown("##### Physical Feature Values (Average)")
                mean_val_b = shap_details['mean_val_before']
                mean_val_a = shap_details['mean_val_after']
                
                # Format Season and Crop Year differently
                val_table = []
                for i, f in enumerate(feat_names):
                    name = readable_names.get(f, f)
                    val_b = mean_val_b[i]
                    val_a = mean_val_a[i]
                    
                    if f == 'Season_encoded':
                        # Season representation (categorical mean is less informative physically, just write "Encoded")
                        val_table.append([name, "Seasonal Class", "Seasonal Class"])
                    elif f == 'Crop_Year':
                        val_table.append([name, f"{int(val_b)}", f"{int(val_a)}"])
                    elif f in ['Area', 'Fertilizer', 'Pesticide']:
                        val_table.append([name, f"{val_b:,.0f}", f"{val_a:,.0f}"])
                    else:
                        val_table.append([name, f"{val_b:,.1f}", f"{val_a:,.1f}"])
                        
                val_df = pd.DataFrame(val_table, columns=["Feature", "Before Segment", "After Segment"])
                st.dataframe(val_df, hide_index=True, use_container_width=True)

# ----------------- TAB 2: TOP 5 NATIONAL FINDINGS -----------------
with tab_top_findings:
    st.subheader("Top 5 Most Statistically Significant Yield Breaks in India")
    st.markdown("Ranked by structural break magnitude in standard deviations (σ) of their series.")
    
    # Sort report data by shift_in_std descending and take top 5
    top_findings = sorted(report_data, key=lambda x: x['shift_in_std'], reverse=True)[:5]
    
    for rank, finding in enumerate(top_findings, 1):
        y_diff = finding['yield_after'] - finding['yield_before']
        y_pct = finding['yield_pct_change']
        shift_class = "highlight-green" if y_diff > 0 else "highlight-red"
        shift_sign = "+" if y_diff > 0 else ""
        
        st.markdown(f"""
        <div class="top-finding-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: #f1f5f9;">Rank {rank}: {finding['state']} — {finding['crop']}</h4>
                <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid #38bdf8; padding: 4px 10px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                    Shift Magnitude: {finding['shift_in_std']:.2f} σ
                </span>
            </div>
            <div style="margin-top: 10px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem; display: block;">Break Year</span>
                    <strong style="color: #cbd5e1; font-size: 1rem;">{finding['break_year']} ({finding['break_season']})</strong>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem; display: block;">Yield Before</span>
                    <strong style="color: #cbd5e1; font-size: 1rem;">{finding['yield_before']:.3f} t/ha</strong>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem; display: block;">Yield After</span>
                    <strong style="color: #cbd5e1; font-size: 1rem;">{finding['yield_after']:.3f} t/ha</strong>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem; display: block;">Yield Shift</span>
                    <strong class="{shift_class}" style="font-size: 1rem;">{shift_sign}{y_diff:.3f} t/ha ({shift_sign}{y_pct:.1f}%)</strong>
                </div>
            </div>
            <div style="margin-top: 12px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 8px;">
                <span style="color: #94a3b8; font-size: 0.85rem; display: block; margin-bottom: 2px;">AI Explanation</span>
                <span style="color: #38bdf8; font-style: italic; font-weight: 500;">"{finding['explanation']}"</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
