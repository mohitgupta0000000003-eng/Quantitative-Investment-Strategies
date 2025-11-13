import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
warnings.filterwarnings("ignore")

# Configure the page
st.set_page_config(
    page_title="Quant Investment Strategies",
    page_icon="📈",
    layout="wide"
)

# Strategy Configuration - This is where you add/remove strategies
STRATEGY_CONFIG = {
    "Momentum500 Basket": {
        "nav_file": r"C:\Users\mohit.gupta\Downloads\Equity_Project\2_Extraction Codes\Strategy\3_Quant_Monthly_Strategy_Momentum\7_NAV Calculator\Portfolio_Daily_Returns_NAV.xlsx",
        "sheet_name": "Stock_NAV"
    },
    # Add more strategies here as needed:
     "LargeCap Basket": {
         "nav_file": r"C:\Users\mohit.gupta\Downloads\Equity_Project\2_Extraction Codes\Strategy\4_Quant_Monthly_Strategy_Momentum_Top100\7_NAV Calculator\Portfolio_Daily_Returns_NAV.xlsx",
         "sheet_name": "Stock_NAV"
    },
    "Midcap Basket": {
         "nav_file": r"C:\Users\mohit.gupta\Downloads\Equity_Project\2_Extraction Codes\Strategy\5_Quant_Monthly_Strategy_Momentum_Mid150\7_NAV Calculator\Portfolio_Daily_Returns_NAV.xlsx", 
         "sheet_name": "Stock_NAV"
     },
     "Large&Midcap Basket": {
         "nav_file": r"C:\Users\mohit.gupta\Downloads\Equity_Project\2_Extraction Codes\Strategy\6_Quant_Monthly_Strategy_Momentum_Top200\7_NAV Calculator\Portfolio_Daily_Returns_NAV.xlsx", 
         "sheet_name": "Stock_NAV"
     },
          "LowVol_Adjusted_Basket": {
         "nav_file": r"C:\Users\mohit.gupta\Downloads\Equity_Project\2_Extraction Codes\Strategy\7_Quant_Monthly_Strategy_Low_Volatility_Adjusted\7_NAV Calculator\Portfolio_Daily_Returns_NAV.xlsx", 
         "sheet_name": "Stock_NAV"
     }
}

# Common data files (same for all strategies)
COMMON_DATA = {
    "price_file": r"C:\Users\mohit.gupta\Downloads\Equity_Project\2_Extraction Codes\Strategy\daily_data.parquet",
    "nifty50_path": r"C:\Users\mohit.gupta\Downloads\Equity_Project\2_Extraction Codes\Strategy\NIFTY_1DAY_2015-01-01_to_2025-11-10.csv",
    "nifty500_path": r"C:\Users\mohit.gupta\Downloads\Equity_Project\2_Extraction Codes\Strategy\NIFTY500_1DAY_2015-01-01_to_2025-11-11.csv",
    "midcap150_path": r"C:\Users\mohit.gupta\Downloads\Equity_Project\2_Extraction Codes\Strategy\NIFTYMIDCAP150_1DAY_2015-01-01_to_2025-11-11.csv",
    "nifty200mo30_path": r"C:\Users\mohit.gupta\Downloads\Equity_Project\2_Extraction Codes\Strategy\NIFTY200Momentum30_1DAY_2015-01-01_to_2025-11-11.csv"
}

@st.cache_data
def load_stock_nav_data(strategy_name):
    """Load the Excel file with portfolio data for a specific strategy"""
    try:
        config = STRATEGY_CONFIG[strategy_name]
        excel_file = config["nav_file"]
        sheet_name = config["sheet_name"]
        
        stock_nav_df = pd.read_excel(excel_file, sheet_name=sheet_name)
        stock_nav_df['Date'] = pd.to_datetime(stock_nav_df['Date'])
        return stock_nav_df
    except Exception as e:
        st.error(f"Error loading Excel file for {strategy_name}: {e}")
        return None

@st.cache_data
def load_share_price_data():
    """Load the parquet file with share prices"""
    try:
        price_df = pd.read_parquet(COMMON_DATA["price_file"])
        
        # Auto-detect columns
        date_col = [col for col in price_df.columns if 'date' in col.lower()][0]
        company_col = [col for col in price_df.columns if 'company' in col.lower() or 'name' in col.lower() or 'symbol' in col.lower()][0]
        price_col = [col for col in price_df.columns if 'close' in col.lower() or 'price' in col.lower()][0]
        
        price_df = price_df.rename(columns={
            date_col: 'Date',
            company_col: 'Company_Name', 
            price_col: 'Close'
        })
        
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        return price_df
    except Exception as e:
        st.error(f"Error loading share price data: {e}")
        return None

@st.cache_data
def load_index_data():
    """Load the four index CSV files"""
    try:
        # Load index data
        nifty50 = pd.read_csv(COMMON_DATA["nifty50_path"])
        nifty500 = pd.read_csv(COMMON_DATA["nifty500_path"])
        midcap150 = pd.read_csv(COMMON_DATA["midcap150_path"])
        nifty200mo30 = pd.read_csv(COMMON_DATA["nifty200mo30_path"])
        
        # Convert timestamp to datetime and extract date
        for df in [nifty50, nifty500, midcap150, nifty200mo30]:
            df['Date'] = pd.to_datetime(df['timestamp']).dt.date
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Create dictionaries for fast index price lookups
        index_data = {
            'Nifty50': dict(zip(nifty50['Date'], nifty50['close'])),
            'Nifty500': dict(zip(nifty500['Date'], nifty500['close'])),
            'Nifty_Midcap_150': dict(zip(midcap150['Date'], midcap150['close'])),
            'Nifty_200_Mo30': dict(zip(nifty200mo30['Date'], nifty200mo30['close']))
        }
        
        return index_data
        
    except Exception as e:
        st.error(f"Error loading index data: {e}")
        return None

def get_rebalance_dates(stock_nav_df):
    """Identify only actual rebalance dates where portfolio stocks change"""
    rebalance_dates = []
    
    # Sort by date
    stock_nav_df = stock_nav_df.sort_values('Date')
    
    # Get stock columns (excluding Date and Total_NAV)
    stock_columns = [col for col in stock_nav_df.columns if col not in ['Date', 'Total_NAV']]
    
    # Always include the first date
    rebalance_dates.append(stock_nav_df.iloc[0]['Date'])
    
    for i in range(1, len(stock_nav_df)):
        current_row = stock_nav_df.iloc[i]
        previous_row = stock_nav_df.iloc[i-1]
        
        # Check if portfolio composition changed (any stock entered or exited)
        portfolio_changed = False
        
        for stock in stock_columns:
            current_inv = current_row[stock]
            prev_inv = previous_row[stock]
            
            # Stock entered (zero to non-zero) or exited (non-zero to zero)
            if (current_inv > 0 and prev_inv == 0) or (current_inv == 0 and prev_inv > 0):
                portfolio_changed = True
                break  # Even one change means rebalance
        
        if portfolio_changed:
            rebalance_dates.append(current_row['Date'])
    
    return sorted(rebalance_dates)

def calculate_index_return(index_data, index_name, start_date, end_date):
    """Calculate index return for a specific period"""
    try:
        # Find the closest available dates in index data
        start_price = None
        end_price = None
        
        # Look for exact date match first, then look for nearby dates
        for date_offset in [0, 1, -1, 2, -2]:  # Check same day, then +/- 1,2 days
            test_start_date = start_date + pd.Timedelta(days=date_offset)
            test_end_date = end_date + pd.Timedelta(days=date_offset)
            
            if test_start_date in index_data[index_name] and test_end_date in index_data[index_name]:
                start_price = index_data[index_name][test_start_date]
                end_price = index_data[index_name][test_end_date]
                break
        
        if start_price is not None and end_price is not None and start_price > 0:
            return ((end_price - start_price) / start_price) * 100
        else:
            return None
    except:
        return None

@st.cache_data
def precompute_strategy_data(_stock_nav_df, _price_df, _rebalance_dates, _index_data, strategy_name):
    """Precompute ALL data for a strategy - both summary and detailed returns"""
    # Create a dictionary for fast price lookups
    price_dict = {}
    for _, row in _price_df.iterrows():
        key = (row['Date'], row['Company_Name'])
        price_dict[key] = row['Close']
    
    # Get stock columns
    stock_columns = [col for col in _stock_nav_df.columns if col not in ['Date', 'Total_NAV']]
    
    all_period_data = []
    all_detailed_returns = {}  # Store detailed returns for each period
    
    # Process each rebalance period
    total_periods = len(_rebalance_dates) - 1
    
    for period_idx in range(total_periods):
        start_date = _rebalance_dates[period_idx]
        end_date = _rebalance_dates[period_idx + 1]
        
        # Get portfolio for start date
        start_data = _stock_nav_df[_stock_nav_df['Date'] == start_date].iloc[0]
        
        period_returns = []
        detailed_returns = []
        valid_stocks = 0
        
        for stock in stock_columns:
            investment = start_data[stock]
            if investment > 0:
                # Fast lookup for entry price
                entry_key = (start_date, stock)
                exit_key = (end_date, stock)
                
                if entry_key in price_dict and exit_key in price_dict:
                    entry_price = price_dict[entry_key]
                    exit_price = price_dict[exit_key]
                    
                    quantity = investment / entry_price
                    exit_value = quantity * exit_price
                    percent_return = ((exit_value - investment) / investment) * 100
                    
                    period_returns.append(percent_return)
                    detailed_returns.append({
                        'Stock': stock,
                        'Entry Price': round(entry_price, 2),
                        'Exit Price': round(exit_price, 2),
                        'Return %': round(percent_return, 2)  # Already 2 decimal places
                    })
                    valid_stocks += 1
        
        if period_returns:
            avg_return = sum(period_returns) / len(period_returns)
            
            # Calculate index returns
            nifty50_return = calculate_index_return(_index_data, 'Nifty50', start_date, end_date)
            nifty500_return = calculate_index_return(_index_data, 'Nifty500', start_date, end_date)
            midcap150_return = calculate_index_return(_index_data, 'Nifty_Midcap_150', start_date, end_date)
            nifty200mo30_return = calculate_index_return(_index_data, 'Nifty_200_Mo30', start_date, end_date)
            
            # Store summary data with index returns
            period_key = f"{start_date.strftime('%d-%m-%y')}_{end_date.strftime('%d-%m-%y')}"
            all_period_data.append({
                'Start Date': start_date.strftime('%d-%m-%y'),
                'End Date': end_date.strftime('%d-%m-%y'),
                'Start_Date_Obj': start_date,  # Store date object for NAV calculation
                'End_Date_Obj': end_date,      # Store date object for NAV calculation
                'Average Return %': round(avg_return, 2),  # 2 decimal places
                'Nifty50 %': round(nifty50_return, 2) if nifty50_return is not None else 'N/A',
                'Nifty500 %': round(nifty500_return, 2) if nifty500_return is not None else 'N/A',
                'Nifty Midcap 150 %': round(midcap150_return, 2) if midcap150_return is not None else 'N/A',
                'Nifty 200 Mo30 %': round(nifty200mo30_return, 2) if nifty200mo30_return is not None else 'N/A',
                'Stocks Count': valid_stocks
            })
            
            # Store detailed data
            all_detailed_returns[period_key] = pd.DataFrame(detailed_returns)
    
    return {
        'summary': pd.DataFrame(all_period_data),
        'detailed': all_detailed_returns
    }

@st.cache_data
def precompute_all_strategies():
    """Precompute data for ALL strategies at once"""
    st.info("🔄 Loading common data files...")
    
    # Load common data once
    price_df = load_share_price_data()
    index_data = load_index_data()
    
    if price_df is None or index_data is None:
        st.error("Failed to load common data files")
        return None
    
    all_strategies_data = {}
    total_strategies = len(STRATEGY_CONFIG)
    
    # Create progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, strategy_name in enumerate(STRATEGY_CONFIG.keys()):
        # Update progress
        progress_percentage = (i) / total_strategies
        progress_bar.progress(progress_percentage)
        status_text.text(f"Precomputing {strategy_name} ({i+1}/{total_strategies})...")
        
        # Load strategy-specific data
        stock_nav_df = load_stock_nav_data(strategy_name)
        if stock_nav_df is None:
            st.warning(f"Failed to load data for {strategy_name}, skipping...")
            continue
        
        # Get rebalance dates
        rebalance_dates = get_rebalance_dates(stock_nav_df)
        
        if len(rebalance_dates) < 2:
            st.warning(f"Not enough rebalance dates for {strategy_name}, skipping...")
            continue
        
        # Precompute strategy data
        strategy_data = precompute_strategy_data(stock_nav_df, price_df, rebalance_dates, index_data, strategy_name)
        all_strategies_data[strategy_name] = strategy_data
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("✅ All strategies precomputed successfully!")
    
    return all_strategies_data

def calculate_nav_series(summary_df, strategy_name):
    """Calculate NAV series starting from 100 for all strategies"""
    nav_data = []
    
    # Start with NAV 100 for all series
    strategy_nav = 100.0
    nifty50_nav = 100.0
    nifty500_nav = 100.0
    midcap150_nav = 100.0
    nifty200mo30_nav = 100.0
    
    # Add initial point
    first_period = summary_df.iloc[0]
    nav_data.append({
        'Date': first_period['Start_Date_Obj'],
        'Period': 'Start',
        f'{strategy_name}_NAV': strategy_nav,
        'Nifty50_NAV': nifty50_nav,
        'Nifty500_NAV': nifty500_nav,
        'Nifty_Midcap_150_NAV': midcap150_nav,
        'Nifty_200_Mo30_NAV': nifty200mo30_nav
    })
    
    # Calculate NAV for each period
    for _, period in summary_df.iterrows():
        # Calculate Strategy NAV
        strategy_return = period['Average Return %']
        strategy_nav = strategy_nav * (1 + strategy_return / 100)
        
        # Calculate Index NAVs (only if data is available)
        if period['Nifty50 %'] != 'N/A':
            nifty50_return = period['Nifty50 %']
            nifty50_nav = nifty50_nav * (1 + nifty50_return / 100)
        
        if period['Nifty500 %'] != 'N/A':
            nifty500_return = period['Nifty500 %']
            nifty500_nav = nifty500_nav * (1 + nifty500_return / 100)
        
        if period['Nifty Midcap 150 %'] != 'N/A':
            midcap150_return = period['Nifty Midcap 150 %']
            midcap150_nav = midcap150_nav * (1 + midcap150_return / 100)
        
        if period['Nifty 200 Mo30 %'] != 'N/A':
            nifty200mo30_return = period['Nifty 200 Mo30 %']
            nifty200mo30_nav = nifty200mo30_nav * (1 + nifty200mo30_return / 100)
        
        nav_data.append({
            'Date': period['End_Date_Obj'],
            'Period': f"{period['Start Date']} to {period['End Date']}",
            f'{strategy_name}_NAV': round(strategy_nav, 2),
            'Nifty50_NAV': round(nifty50_nav, 2) if period['Nifty50 %'] != 'N/A' else None,
            'Nifty500_NAV': round(nifty500_nav, 2) if period['Nifty500 %'] != 'N/A' else None,
            'Nifty_Midcap_150_NAV': round(midcap150_nav, 2) if period['Nifty Midcap 150 %'] != 'N/A' else None,
            'Nifty_200_Mo30_NAV': round(nifty200mo30_nav, 2) if period['Nifty 200 Mo30 %'] != 'N/A' else None
        })
    
    return pd.DataFrame(nav_data)

def calculate_performance_metrics(nav_df, series_name):
    """Calculate comprehensive performance metrics for a series"""
    if series_name not in nav_df.columns:
        return None
    
    series_data = nav_df[['Date', series_name]].dropna()
    if len(series_data) < 2:
        return None
    
    # Calculate periodic returns
    series_data = series_data.sort_values('Date')
    series_data['Return'] = series_data[series_name].pct_change()
    returns = series_data['Return'].dropna()
    
    if len(returns) == 0:
        return None
    
    # Basic metrics
    total_return = (series_data[series_name].iloc[-1] / series_data[series_name].iloc[0] - 1) * 100
    
    # Calculate years for CAGR
    start_date = series_data['Date'].iloc[0]
    end_date = series_data['Date'].iloc[-1]
    years = (end_date - start_date).days / 365.25
    
    # CAGR
    cagr = ((series_data[series_name].iloc[-1] / series_data[series_name].iloc[0]) ** (1/years) - 1) * 100
    
    # Annual volatility (standard deviation of monthly returns annualized)
    volatility = returns.std() * np.sqrt(12) * 100
    
    # Sharpe Ratio (using 6% risk-free rate as requested)
    risk_free_rate = 0.06
    sharpe = (cagr/100 - risk_free_rate) / (volatility/100) if volatility > 0 else 0
    
    # Maximum Drawdown
    nav_series = series_data[series_name].values
    peak = np.maximum.accumulate(nav_series)
    drawdown = (nav_series - peak) / peak
    max_drawdown = np.min(drawdown) * 100
    
    # Calmar Ratio
    calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
    
    # Calculate annual returns based on calendar years
    annual_returns = {}
    
    # Get the date range
    start_year = start_date.year
    end_year = end_date.year
    
    for year in range(start_year, end_year + 1):
        # Find NAV at the start of the year (Jan 1st or closest available date)
        year_start = pd.Timestamp(f'{year}-01-01')
        start_nav_row = series_data[series_data['Date'] >= year_start].head(1)
        
        # Find NAV at the end of the year (Dec 31st or closest available date)
        year_end = pd.Timestamp(f'{year}-12-31')
        end_nav_row = series_data[series_data['Date'] <= year_end].tail(1)
        
        if len(start_nav_row) > 0 and len(end_nav_row) > 0:
            year_start_nav = start_nav_row[series_name].iloc[0]
            year_end_nav = end_nav_row[series_name].iloc[0]
            
            # Only calculate if we have valid start and end NAVs
            if year_start_nav > 0 and year_end_nav > 0:
                annual_return = (year_end_nav / year_start_nav - 1) * 100
                annual_returns[str(year)] = annual_return
    
    return {
        'Total Return (%)': round(total_return, 2),
        'CAGR (%)': round(cagr, 2),
        'Annual Volatility (%)': round(volatility, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Calmar Ratio': round(calmar, 2),
        'Annual Returns': annual_returns
    }

def create_nav_chart(nav_df, selected_series, strategy_name):
    """Create an interactive NAV chart for selected series"""
    fig = go.Figure()
    
    colors = {
        f'{strategy_name}_NAV': '#1f77b4',
        'Nifty50_NAV': '#ff7f0e', 
        'Nifty500_NAV': '#2ca02c',
        'Nifty_Midcap_150_NAV': '#d62728',
        'Nifty_200_Mo30_NAV': '#9467bd'
    }
    
    names = {
        f'{strategy_name}_NAV': strategy_name,
        'Nifty50_NAV': 'Nifty50',
        'Nifty500_NAV': 'Nifty500',
        'Nifty_Midcap_150_NAV': 'Nifty Midcap 150',
        'Nifty_200_Mo30_NAV': 'Nifty 200 Mo30'
    }
    
    for series in selected_series:
        if series in nav_df.columns and nav_df[series].notna().any():
            fig.add_trace(go.Scatter(
                x=nav_df['Date'],
                y=nav_df[series],
                mode='lines+markers',
                name=names[series],
                line=dict(color=colors[series], width=3),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title=f'{strategy_name} - NAV Performance Comparison (Starting from 100)',
        xaxis_title='Date',
        yaxis_title='NAV Value',
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def color_negative_red(val):
    """Color negative values red and positive values green"""
    if isinstance(val, (int, float)):
        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
        return f'color: {color}'
    elif isinstance(val, str) and val != 'N/A':
        try:
            num_val = float(val)
            color = 'green' if num_val > 0 else 'red' if num_val < 0 else 'black'
            return f'color: {color}'
        except:
            return ''
    return ''

def strategy_dashboard(strategy_name, precomputed_data):
    """Main dashboard for a specific strategy using precomputed data"""
    # Title and navigation
    col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
    with col_header1:
        st.title(f"📈 {strategy_name} (Monthly Rebalance)")
    with col_header2:
        rebalance_dates = list(precomputed_data['detailed'].keys())
        st.metric("Rebalance Periods", len(rebalance_dates))
    with col_header3:
        if st.button("🔄 Switch Strategy"):
            st.session_state.selected_strategy = None
            st.rerun()
    
    # Navigation - Performance Analysis first, then Summary, then Detailed
    view_option = st.radio("Select View:", 
                          ["📈 Performance Analysis", "📊 Summary - All Periods", "🔍 Detailed - Single Period"])
    
    if view_option == "📈 Performance Analysis":
        st.subheader("📈 Performance Analysis")
        
        all_returns_df = precomputed_data['summary']
        
        if not all_returns_df.empty:
            # Calculate NAV series
            nav_df = calculate_nav_series(all_returns_df, strategy_name)
            
            # Series selection for chart
            available_series = [f'{strategy_name}_NAV']
            if nav_df['Nifty50_NAV'].notna().any():
                available_series.append('Nifty50_NAV')
            if nav_df['Nifty500_NAV'].notna().any():
                available_series.append('Nifty500_NAV')
            if nav_df['Nifty_Midcap_150_NAV'].notna().any():
                available_series.append('Nifty_Midcap_150_NAV')
            if nav_df['Nifty_200_Mo30_NAV'].notna().any():
                available_series.append('Nifty_200_Mo30_NAV')
            
            # Series selection
            selected_series = st.multiselect(
                "Select series to display:",
                options=available_series,
                default=[f'{strategy_name}_NAV', 'Nifty50_NAV'],
                format_func=lambda x: {
                    f'{strategy_name}_NAV': strategy_name,
                    'Nifty50_NAV': 'Nifty50',
                    'Nifty500_NAV': 'Nifty500',
                    'Nifty_Midcap_150_NAV': 'Nifty Midcap 150',
                    'Nifty_200_Mo30_NAV': 'Nifty 200 Mo30'
                }[x]
            )
            
            if not selected_series:
                st.warning("Please select at least one series to display")
                return
            
            # 1. Performance Metrics Section (FIRST)
            st.subheader("📊 Performance Metrics")
            
            # Calculate metrics for each selected series
            metrics_data = []
            for series in selected_series:
                metrics = calculate_performance_metrics(nav_df, series)
                if metrics:
                    metrics_row = {'Series': {
                        f'{strategy_name}_NAV': strategy_name,
                        'Nifty50_NAV': 'Nifty50',
                        'Nifty500_NAV': 'Nifty500',
                        'Nifty_Midcap_150_NAV': 'Nifty Midcap 150',
                        'Nifty_200_Mo30_NAV': 'Nifty 200 Mo30'
                    }[series]}
                    metrics_row.update({k: v for k, v in metrics.items() if k != 'Annual Returns'})
                    metrics_data.append(metrics_row)
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df = metrics_df.set_index('Series')
                
                # Display metrics table
                st.dataframe(metrics_df.style.format({
                    'Total Return (%)': '{:.2f}%',
                    'CAGR (%)': '{:.2f}%',
                    'Annual Volatility (%)': '{:.2f}%',
                    'Sharpe Ratio': '{:.2f}',
                    'Max Drawdown (%)': '{:.2f}%',
                    'Calmar Ratio': '{:.2f}'
                }), use_container_width=True)
            
            # 2. Annual Returns Section (SECOND)
            st.subheader("📅 Annual Returns")
            
            # Calculate and display annual returns for each selected series
            annual_returns_data = []
            for series in selected_series:
                metrics = calculate_performance_metrics(nav_df, series)
                if metrics and 'Annual Returns' in metrics:
                    annual_returns = metrics['Annual Returns']
                    row = {'Year': {
                        f'{strategy_name}_NAV': strategy_name,
                        'Nifty50_NAV': 'Nifty50',
                        'Nifty500_NAV': 'Nifty500',
                        'Nifty_Midcap_150_NAV': 'Nifty Midcap 150',
                        'Nifty_200_Mo30_NAV': 'Nifty 200 Mo30'
                    }[series]}
                    row.update(annual_returns)
                    annual_returns_data.append(row)
            
            if annual_returns_data:
                annual_df = pd.DataFrame(annual_returns_data)
                annual_df = annual_df.set_index('Year')
                
                # Apply color coding to annual returns
                styled_annual_df = annual_df.style.format('{:.2f}%').applymap(color_negative_red)
                st.dataframe(styled_annual_df, use_container_width=True)
            
            # 3. NAV Chart Section (THIRD)
            st.subheader("📈 NAV Performance Chart")
            nav_chart = create_nav_chart(nav_df, selected_series, strategy_name)
            st.plotly_chart(nav_chart, use_container_width=True)
            
            # 4. NAV Data Table (FOURTH - at the bottom)
            st.subheader("NAV Series Data")
            display_nav_df = nav_df[['Date', 'Period'] + selected_series].copy()
            display_nav_df['Date'] = display_nav_df['Date'].dt.strftime('%d-%m-%y')
            
            # Rename columns for display
            column_mapping = {
                f'{strategy_name}_NAV': strategy_name,
                'Nifty50_NAV': 'Nifty50',
                'Nifty500_NAV': 'Nifty500',
                'Nifty_Midcap_150_NAV': 'Nifty Midcap 150',
                'Nifty_200_Mo30_NAV': 'Nifty 200 Mo30'
            }
            display_nav_df = display_nav_df.rename(columns=column_mapping)
            
            # Only format numeric columns, leave string columns alone
            numeric_columns = [col for col in display_nav_df.columns if col in [strategy_name, 'Nifty50', 'Nifty500', 'Nifty Midcap 150', 'Nifty 200 Mo30']]
            
            if numeric_columns:
                styled_nav_df = display_nav_df.style.format({col: '{:.2f}' for col in numeric_columns})
            else:
                styled_nav_df = display_nav_df.style
            
            st.dataframe(styled_nav_df, use_container_width=True, height=400)
    
    elif view_option == "📊 Summary - All Periods":
        st.subheader("📊 All Rebalance Period Returns")
        
        all_returns_df = precomputed_data['summary']
        
        if not all_returns_df.empty:
            # Display summary statistics
            total_periods = len(all_returns_df)
            overall_avg_return = all_returns_df['Average Return %'].mean()
            positive_periods = len(all_returns_df[all_returns_df['Average Return %'] > 0])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Periods", total_periods)
            with col2:
                st.metric(f"{strategy_name} Avg Return", f"{overall_avg_return:.2f}%")
            with col3:
                st.metric("Positive Periods", f"{positive_periods}/{total_periods}")
            
            # Apply color coding to the summary table
            styled_summary_df = all_returns_df[['Start Date', 'End Date', 'Average Return %', 'Nifty50 %', 'Nifty500 %', 'Nifty Midcap 150 %', 'Nifty 200 Mo30 %', 'Stocks Count']].style.format({
                'Average Return %': '{:.2f}%',
                'Nifty50 %': '{:.2f}%',
                'Nifty500 %': '{:.2f}%',
                'Nifty Midcap 150 %': '{:.2f}%',
                'Nifty 200 Mo30 %': '{:.2f}%'
            }).applymap(color_negative_red, subset=[
                'Average Return %', 'Nifty50 %', 'Nifty500 %', 'Nifty Midcap 150 %', 'Nifty 200 Mo30 %'
            ])
            
            # Display the styled summary table
            st.dataframe(styled_summary_df, use_container_width=True)
            
            # Add some basic analysis
            st.subheader("📈 Performance Analysis")
            best_period = all_returns_df.loc[all_returns_df['Average Return %'].idxmax()]
            worst_period = all_returns_df.loc[all_returns_df['Average Return %'].idxmin()]
            
            col4, col5 = st.columns(2)
            with col4:
                st.metric("Best Period", 
                         f"{best_period['Average Return %']:.2f}%",
                         f"{best_period['Start Date']} to {best_period['End Date']}")
            with col5:
                st.metric("Worst Period", 
                         f"{worst_period['Average Return %']:.2f}%",
                         f"{worst_period['Start Date']} to {worst_period['End Date']}")
    
    else:  # Detailed - Single Period view
        st.subheader("🔍 Detailed Period Analysis")
        
        # Get rebalance dates from precomputed data - FIXED: Sort chronologically
        rebalance_periods = list(precomputed_data['detailed'].keys())
        
        # Extract unique start dates and convert to datetime for proper sorting
        rebalance_dates_display = []
        date_objects = []
        
        for period_key in rebalance_periods:
            start_display = period_key.split('_')[0]
            # Convert display string to datetime object for sorting
            date_obj = datetime.strptime(start_display, '%d-%m-%y')
            rebalance_dates_display.append(start_display)
            date_objects.append(date_obj)
        
        # Sort by datetime objects but keep display strings
        sorted_indices = np.argsort(date_objects)
        rebalance_dates_display = [rebalance_dates_display[i] for i in sorted_indices]
        
        # Create mapping for period lookup
        rebalance_dates_dict = {}
        for period_key in rebalance_periods:
            start_display = period_key.split('_')[0]
            rebalance_dates_dict[start_display] = period_key
        
        # Date selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_start_display = st.selectbox(
                "Start Date (Rebalance):",
                options=rebalance_dates_display[:-1],
                index=len(rebalance_dates_display)-2 if len(rebalance_dates_display) > 1 else 0
            )
        
        with col2:
            start_index = rebalance_dates_display.index(selected_start_display)
            available_end_dates = rebalance_dates_display[start_index + 1:]
            
            selected_end_display = st.selectbox(
                "End Date (Next Rebalance):",
                options=available_end_dates,
                index=0
            )
        
        # Get precomputed detailed returns (INSTANT)
        period_key = f"{selected_start_display}_{selected_end_display}"
        returns_df = precomputed_data['detailed'].get(period_key, pd.DataFrame())
        
        if not returns_df.empty:
            avg_return = returns_df['Return %'].mean()
            
            # Get index returns for this period from summary data
            summary_row = precomputed_data['summary'][
                (precomputed_data['summary']['Start Date'] == selected_start_display) & 
                (precomputed_data['summary']['End Date'] == selected_end_display)
            ]
            
            st.subheader(f"Returns from {selected_start_display} to {selected_end_display}")
            
            # Display metrics including index comparisons
            col_avg1, col_avg2, col_avg3, col_avg4, col_avg5, col_avg6 = st.columns(6)
            with col_avg1:
                st.metric(f"{strategy_name} Return", f"{avg_return:.2f}%")
            with col_avg2:
                st.metric("Portfolio Size", f"{len(returns_df)} stocks")
            with col_avg3:
                nifty50_val = summary_row['Nifty50 %'].iloc[0] if not summary_row.empty else 'N/A'
                st.metric("Nifty50", f"{nifty50_val}%" if nifty50_val != 'N/A' else "N/A")
            with col_avg4:
                nifty500_val = summary_row['Nifty500 %'].iloc[0] if not summary_row.empty else 'N/A'
                st.metric("Nifty500", f"{nifty500_val}%" if nifty500_val != 'N/A' else "N/A")
            with col_avg5:
                midcap_val = summary_row['Nifty Midcap 150 %'].iloc[0] if not summary_row.empty else 'N/A'
                st.metric("Nifty Midcap 150", f"{midcap_val}%" if midcap_val != 'N/A' else "N/A")
            with col_avg6:
                nifty200mo30_val = summary_row['Nifty 200 Mo30 %'].iloc[0] if not summary_row.empty else 'N/A'
                st.metric("Nifty 200 Mo30", f"{nifty200mo30_val}%" if nifty200mo30_val != 'N/A' else "N/A")
            
            # Add serial numbers starting from 1
            returns_df_display = returns_df.copy()
            returns_df_display = returns_df_display.reset_index(drop=True)
            returns_df_display.index = returns_df_display.index + 1
            returns_df_display.index.name = 'S.No.'
            
            # Apply color coding to the detailed table
            styled_detailed_df = returns_df_display.style.format({
                'Entry Price': '₹{:.2f}',
                'Exit Price': '₹{:.2f}',
                'Return %': '{:.2f}%'
            }).applymap(color_negative_red, subset=['Return %'])
            
            # Display the styled table
            st.dataframe(styled_detailed_df, use_container_width=True, height=400)
            
        else:
            st.warning("No data found for selected rebalance period")

def main():
    # Initialize session state
    if 'selected_strategy' not in st.session_state:
        st.session_state.selected_strategy = None
    if 'all_precomputed_data' not in st.session_state:
        st.session_state.all_precomputed_data = None
    
    # Precompute all strategies data if not already done
    if st.session_state.all_precomputed_data is None:
        st.session_state.all_precomputed_data = precompute_all_strategies()
        
        # If precomputation failed, show error
        if st.session_state.all_precomputed_data is None:
            st.error("Failed to precompute strategy data. Please check your data files and paths.")
            return
        
        # Clear the loading messages and rerun to show the strategy selection
        st.rerun()
    
    # Strategy Selection Page
    if st.session_state.selected_strategy is None:
        st.title("📊 Quant Investment Strategies")
        st.markdown("### Select a Strategy to Analyze")
        
        # Display available strategies
        strategies = list(STRATEGY_CONFIG.keys())
        
        if not strategies:
            st.error("No strategies configured. Please check STRATEGY_CONFIG.")
            return
        
        col1, col2, col_3 = st.columns([1, 2, 1])
        with col2:
            selected = st.selectbox(
                "Choose Strategy:",
                options=strategies,
                index=0
            )
            
            if st.button("🚀 Analyze Strategy"):
                st.session_state.selected_strategy = selected
                st.rerun()
        
        # Display strategy information
        st.markdown("---")
        st.subheader("📋 Available Strategies")
        for strategy in strategies:
            with st.expander(f"🔹 {strategy}"):
                st.write(f"**File:** {STRATEGY_CONFIG[strategy]['nav_file']}")
                st.write(f"**Sheet:** {STRATEGY_CONFIG[strategy]['sheet_name']}")
                if strategy in st.session_state.all_precomputed_data:
                    periods = len(st.session_state.all_precomputed_data[strategy]['summary'])
                    st.write(f"**Precomputed Periods:** {periods}")
                else:
                    st.write("❌ **Status:** Data not available")
        
        # Disclaimer Section
        st.markdown("---")
        st.subheader("📝 Important Disclaimers")
        
        disclaimer_expander = st.expander("⚠️ Click to view backtesting methodology and limitations", expanded=True)
        with disclaimer_expander:
            st.markdown("""
            **Backtesting Methodology & Limitations:**
            
            - **Gross Basis Calculation**: Back-test results are presented on a gross basis. Transaction costs, taxes, and other charges could negatively impact overall CAGR by approximately **1-3%**.
            
            - **Perfect Allocation Assumption**: Analysis assumes perfect fractional share allocation. In reality, discrete share quantities may reduce CAGR by approximately **2-3%**.
            
            - **Static Universe**: Back-test uses Nifty 500 stock composition as of **30th June 2025**. Historical changes in index constituents are not reflected, which may impact CAGR by approximately **2-3%**.
            
            - **Equal Weighting**: Returns are calculated using equal monthly capital allocation. Market fluctuations and profit/loss realization in practice may affect CAGR by approximately **2-3%**.
            
            - **Dividend Exclusion**: Dividend income has not been considered in returns calculation. Including dividends would likely provide a positive impact of approximately **1%** on CAGR.
            
            - **Data Source**: Historical adjusted prices sourced via **Dhan paid API**.
            
            **Overall Impact Assessment:**
            The above factors partially offset each other (both positively and negatively). The net likely impact on reported CAGR figures is approximately **3-5%**.
            
            **Note**: These results are indicative and represent a Minimum Viable Product (MVP) stage analysis. 
            """)

    
    # Strategy Dashboard
    else:
        strategy_name = st.session_state.selected_strategy
        if strategy_name in st.session_state.all_precomputed_data:
            strategy_dashboard(strategy_name, st.session_state.all_precomputed_data[strategy_name])
        else:
            st.error(f"No precomputed data available for {strategy_name}")
            if st.button("← Back to Strategy Selection"):
                st.session_state.selected_strategy = None
                st.rerun()

if __name__ == "__main__":
    main()