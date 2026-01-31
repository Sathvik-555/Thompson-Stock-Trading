import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from datetime import datetime, timedelta
from thompson_trader import (
    ThompsonSamplingStockTrader,
    portfolio1,
    portfolio2,
    run_multiple_simulations,
    download_and_prepare_data
)

st.title("📈 Thompson Sampling Stock Trader Dashboard")

# Sidebar
st.sidebar.header("Simulation Settings")
num_simulations = st.sidebar.slider("Number of Simulations", 10, 500, 100)
seed = st.sidebar.number_input("Random Seed", value=42)
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.today())

# Cache stock data
@st.cache_data(show_spinner="Downloading stock data...")
def get_stock_data(portfolio, start, end):
    return download_and_prepare_data(portfolio, start, end)

# Download data once
st.write("📥 Loading stock data...")
data1, stats1 = get_stock_data(portfolio1, start_date, end_date)
data2, stats2 = get_stock_data(portfolio2, start_date, end_date)

# Filter portfolios to only valid symbols
valid_symbols1 = [s for s in portfolio1 if s in stats1.index]
valid_symbols2 = [s for s in portfolio2 if s in stats2.index]

# Optional: Warn if any stocks are missing
removed1 = set(portfolio1) - set(valid_symbols1)
removed2 = set(portfolio2) - set(valid_symbols2)

if removed1:
    st.warning(f"⚠️ The following Large-cap stocks were removed due to missing data: {', '.join(removed1)}")
if removed2:
    st.warning(f"⚠️ The following Top Performers were removed due to missing data: {', '.join(removed2)}")

# Run simulations
st.write("🏃 Running simulations...")
avg1, std1, mean_ret1, std_ret1, mean_shp1, std_shp1, selections1 = run_multiple_simulations(
    ThompsonSamplingStockTrader, portfolio1, data1, stats1, num_simulations, seed
)
avg2, std2, mean_ret2, std_ret2, mean_shp2, std_shp2, selections2 = run_multiple_simulations(
    ThompsonSamplingStockTrader, portfolio2, data2, stats2, num_simulations, seed
)


# Display results
st.subheader("📊 Simulation Results")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏢 Large-cap Portfolio")
    st.metric("Mean Total Return", f"{mean_ret1:.2f} %")
    st.metric("Mean Sharpe Ratio", f"{mean_shp1:.2f}")

with col2:
    st.markdown("### 🚀 Top Performers Portfolio")
    st.metric("Mean Total Return", f"{mean_ret2:.2f} %")
    st.metric("Mean Sharpe Ratio", f"{mean_shp2:.2f}")

# Altair line chart
st.subheader("📈 Portfolio Value Over Time")

df_plot = pd.DataFrame({
    'Day': np.arange(len(avg1)),
    'Large-cap (mean)': avg1,
    'Top Performers (mean)': avg2,
    'Large-cap (low)': avg1 - std1,
    'Large-cap (high)': avg1 + std1,
    'Top Performers (low)': avg2 - std2,
    'Top Performers (high)': avg2 + std2
})

line_chart = alt.Chart(df_plot.reset_index()).transform_fold(
    ['Large-cap (mean)', 'Top Performers (mean)'],
    as_=['Portfolio', 'Value']
).mark_line().encode(
    x='Day:Q',
    y='Value:Q',
    color='Portfolio:N'
)

band1 = alt.Chart(df_plot).mark_area(opacity=0.2).encode(
    x='Day:Q',
    y='Large-cap (low):Q',
    y2='Large-cap (high):Q'
)
band2 = alt.Chart(df_plot).mark_area(opacity=0.2).encode(
    x='Day:Q',
    y='Top Performers (low):Q',
    y2='Top Performers (high):Q'
)

st.altair_chart((line_chart + band1 + band2).interactive(), use_container_width=True)

# Selection frequency visualization
st.subheader("📌 Most Frequently Selected Stocks")

# Count selections
sel_count1 = pd.Series(selections1).value_counts().sort_values(ascending=False)
sel_count2 = pd.Series(selections2).value_counts().sort_values(ascending=False)

# Top 10
top1 = sel_count1.head(10).reset_index()
top1.columns = ['Stock', 'Count']
top2 = sel_count2.head(10).reset_index()
top2.columns = ['Stock', 'Count']

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏢 Large-cap Portfolio")
    bar1 = alt.Chart(top1).mark_bar().encode(
        x=alt.X('Count:Q'),
        y=alt.Y('Stock:N', sort='-x')
    )
    st.altair_chart(bar1, use_container_width=True)

with col2:
    st.markdown("### 🚀 Top Performers Portfolio")
    bar2 = alt.Chart(top2).mark_bar().encode(
        x=alt.X('Count:Q'),
        y=alt.Y('Stock:N', sort='-x')
    )
    st.altair_chart(bar2, use_container_width=True)
