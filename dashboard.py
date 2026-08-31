import streamlit as st
import sqlite3
import pandas as pd
import json
import time
import altair as alt
from datetime import datetime

st.set_page_config(layout="wide", page_title="Razorpay AI Risk Agent")

# CSS for action colors
st.markdown("""
<style>
    .action-pass { color: #28a745; font-weight: bold; }
    .action-step-up { color: #ffc107; font-weight: bold; }
    .action-block { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

DB_PATH = "audit_ledger.db"

def get_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM audit_trail ORDER BY timestamp DESC LIMIT 10", conn)
        conn.close()
        return df
    except sqlite3.Error:
        return pd.DataFrame()

def check_circuit_breaker(df):
    """Simple check based on last 20 txns block rate, mimicking the app's internal logic"""
    try:
        conn = sqlite3.connect(DB_PATH)
        recent_df = pd.read_sql_query("SELECT action FROM audit_trail ORDER BY timestamp DESC LIMIT 20", conn)
        conn.close()
        if len(recent_df) >= 20:
            blocks = len(recent_df[recent_df['action'] == 'ACTION_BLOCK'])
            if blocks / len(recent_df) > 0.08:
                return "🔴 TRIPPED"
        return "🟢 ONLINE"
    except:
        return "🟢 ONLINE"

def display_dashboard():
    st.title("🛡️ Fraud Risk Agent Dashboard (Real-Time)")
    
    # Refresh logic
    placeholder = st.empty()
    
    with placeholder.container():
        df = get_data()
        
        if df.empty:
            st.warning("Waiting for transactions... No data in audit_ledger.db yet.")
            time.sleep(2)
            st.rerun()

        cb_status = check_circuit_breaker(df)
        st.subheader(f"Circuit Breaker Status: {cb_status}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Live Transaction Stream")
            
            # Format dataframe for display
            display_df = df[['tx_id', 'amount', 'risk_score', 'action', 'latency_ms']].copy()
            display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:.2f}")
            display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.4f}")
            display_df['latency_ms'] = display_df['latency_ms'].apply(lambda x: f"{x:.1f} ms")
            
            # Apply color formatting
            def color_actions(val):
                if val == "ACTION_PASS": return 'color: #28a745; font-weight: bold'
                elif val == "ACTION_STEP_UP": return 'color: #ffc107; font-weight: bold'
                elif val == "ACTION_BLOCK": return 'color: #dc3545; font-weight: bold'
                return ''
                
            st.dataframe(
                display_df.style.map(color_actions, subset=['action']),
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.subheader("Latest Block Reasons (SHAP)")
            
            # Find most recent block
            blocks = df[df['action'] == 'ACTION_BLOCK']
            if not blocks.empty:
                latest_block = blocks.iloc[0]
                reasons_str = latest_block['reasons']
                try:
                    reasons = json.loads(reasons_str)
                    if reasons:
                        # Convert to DataFrame for Altair
                        chart_data = pd.DataFrame(reasons)
                        # Assume reasons have 'feature', 'shap_value', 'feature_value'
                        chart = alt.Chart(chart_data).mark_bar().encode(
                            x=alt.X('shap_value:Q', title="Impact (SHAP Value)"),
                            y=alt.Y('feature:N', sort='-x', title="Feature"),
                            color=alt.condition(
                                alt.datum.shap_value > 0,
                                alt.value("#dc3545"),  # Red for increasing risk
                                alt.value("#28a745")   # Green for decreasing risk
                            ),
                            tooltip=['feature', 'feature_value', 'shap_value']
                        ).properties(height=300)
                        
                        st.altair_chart(chart, use_container_width=True)
                        st.caption(f"Transaction: {latest_block['tx_id']}")
                    else:
                        st.info("No explainability data found for this block.")
                except:
                    st.info("Could not parse SHAP data.")
            else:
                st.info("No blocked transactions recently.")
                
    time.sleep(2)
    st.rerun()

if __name__ == "__main__":
    display_dashboard()
