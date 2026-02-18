
# Streamlit Application Specification: AI-Enhanced Portfolio Risk & Sentiment Dashboard

## 1. Application Overview

### Purpose of the Application

This Streamlit application serves as an "AI-Enhanced Portfolio Risk & Sentiment Dashboard" designed specifically for **CFA Charterholders and Investment Professionals** at **Apex Capital Management**. The primary goal is to address the "last mile problem" of AI adoption by seamlessly integrating AI-generated financial insights—such as sentiment scores, credit default probabilities, and ESG governance details—into familiar analytical workflows and reporting tools. The application aims to provide a dynamic, interactive platform for daily portfolio monitoring, enhancing decision-making with proactive signals, and ensuring robust compliance.

### High-Level Story Flow

1.  **Welcome & Overview:** The user (a portfolio manager) lands on the application, greeted by an overview explaining the value proposition of AI integration in their daily work. This sets the stage for a practical, workflow-driven experience.
2.  **Portfolio Sentiment Analysis:** The user navigates to the "Portfolio Sentiment" page to quickly assess the market sentiment for their portfolio holdings. They observe AI-generated sentiment scores and trends, visually highlighted for immediate attention to positive or negative shifts. This helps in understanding market perception and making timely adjustments.
3.  **Credit Risk Insights:** Moving to "Credit Risk Analysis," the user reviews AI-derived credit default probabilities and their corresponding implied credit ratings for corporate obligors. Conditional styling provides quick visual cues on credit health, enabling rapid risk identification.
4.  **ESG Governance Deep Dive:** The "ESG Governance Report" allows the user to examine RAG-extracted governance details, such as the linkage of ESG metrics to CEO compensation. This provides deeper insights into non-financial risks and opportunities, crucial for ESG-mandated portfolios or committee presentations.
5.  **Automated Report Generation:** On the "Report Generation" page, the user can trigger the creation of professional, AI-enhanced Excel dashboards (for detailed analysis) and interactive HTML dashboards (for high-level overviews) or templated HTML reports (for consistent communication). This simulates the output of an automated reporting pipeline, ready for internal committees or client discussions.
6.  **Compliance & Model Transparency:** Throughout the application, and particularly in a dedicated "Compliance & Model Info" section, the user finds critical metadata about the AI models, data sources, and compliance flags. This ensures transparency and adherence to ethical guidelines, reinforcing the responsible use of AI in finance.

This flow mirrors a real-world scenario where an investment professional leverages AI outputs to streamline reporting, enhance analytical capabilities, and maintain regulatory standards, all within an intuitive interface.

## 2. Code Requirements

### Import Statement

```python
from source import *
import streamlit as st
import pandas as pd
import os
import base64
from datetime import datetime
```

### `st.session_state` Initialization, Updates, and Reads

`st.session_state` will be extensively used to maintain the application's state across user interactions and page navigations.

**Initialization (at the beginning of `app.py` before any UI rendering):**

```python
if 'page' not in st.session_state:
    st.session_state.page = 'Overview'

# Initialize dataframes and report paths only once
if 'current_sentiment_df' not in st.session_state:
    st.session_state.current_sentiment_df, \
    st.session_state.sentiment_time_series_df, \
    st.session_state.credit_df, \
    st.session_state.esg_governance_df, \
    st.session_state.esg_overall_df = generate_synthetic_data()

if 'excel_dashboard_path' not in st.session_state:
    st.session_state.excel_dashboard_path = None
if 'esg_report_path' not in st.session_state:
    st.session_state.esg_report_path = None
if 'html_dashboard_path' not in st.session_state:
    st.session_state.html_dashboard_path = None
if 'html_report_path' not in st.session_state:
    st.session_state.html_report_path = None
```

**Updates:**

*   `st.session_state.page`: Updated when the user selects a different page from the sidebar `st.selectbox`.
*   `st.session_state.excel_dashboard_path`: Updated by the `create_ai_dashboard_excel` function call after generation.
*   `st.session_state.esg_report_path`: Updated by the `create_esg_governance_report` function call after generation.
*   `st.session_state.html_dashboard_path`: Updated by the `create_interactive_dashboard` function call after generation.
*   `st.session_state.html_report_path`: Updated by the `generate_html_report_from_template` function call after generation.

**Reads (across pages):**

*   All dataframes (`current_sentiment_df`, `sentiment_time_series_df`, `credit_df`, `esg_governance_df`, `esg_overall_df`) are read from `st.session_state` by the respective page rendering logic to display data and generate reports.
*   All report paths (`excel_dashboard_path`, `esg_report_path`, `html_dashboard_path`, `html_report_path`) are read to provide download links for the generated files.

### UI Interactions and `source.py` Function Invocations

The application will use a sidebar `st.selectbox` for navigation. Conditional rendering based on `st.session_state.page` will display the content for each "page".

**Sidebar Navigation:**

```python
with st.sidebar:
    st.title("Navigation")
    st.session_state.page = st.selectbox(
        "Choose a section",
        ['Overview', 'Portfolio Sentiment', 'Credit Risk Analysis', 'ESG Governance Report', 'Report Generation', 'Compliance & Model Info']
    )
```

---

### Page: Overview

**Markdown:**

```python
st.title("AI-Enhanced Portfolio Risk & Sentiment Dashboard")
st.markdown("---")
st.markdown(f"")
st.markdown(f"As a **CFA Charterholder and Investment Professional** at **Apex Capital Management**, my daily mandate involves meticulously monitoring our investment portfolios. In today's fast-paced markets, staying ahead requires not just deep financial acumen but also leveraging cutting-edge technology. My biggest challenge is to quickly assimilate vast amounts of market data and forward-looking insights, like market sentiment and credit risk, into actionable intelligence. Traditionally, this meant sifting through countless reports and manually aggregating data, a time-consuming process prone to delays.")
st.markdown(f"")
st.markdown(f"This dashboard demonstrates a critical workflow: how we integrate AI-generated insights directly into our familiar tools—Excel dashboards, interactive HTML reports, and this Streamlit application. This bridges the **'last mile problem'** of AI adoption, ensuring that sophisticated AI model outputs are not confined to technical environments but are immediately accessible and actionable for daily portfolio monitoring and critical decision-making. By automating the integration of AI-powered sentiment analysis and credit risk assessments, we can streamline our workflow, enhance our decision-making with proactive signals, and maintain robust compliance.")
st.markdown(f"")
st.markdown(f"---")
st.header("Key Insights Delivered by AI:")
st.markdown(f"- **AI-Generated Sentiment Scores**: Quickly gauge market sentiment for portfolio holdings.")
st.markdown(f"- **AI-Derived Credit Default Probabilities**: Assess credit risk for obligors with implied credit ratings.")
st.markdown(f"- **RAG-Extracted ESG Governance**: Understand non-financial risks from regulatory filings.")
st.markdown(f"- **Automated Reporting**: Generate professional Excel and HTML reports for internal and external stakeholders.")
st.markdown(f"")
st.info("Use the sidebar to navigate through the AI-enhanced insights and reports.")
```

**Function Calls:** None directly on this page; it serves as an introduction. Initial data generation `generate_synthetic_data()` happens on app startup.

---

### Page: Portfolio Sentiment

**Markdown:**

```python
st.header("Portfolio Sentiment Analysis")
st.markdown(f"")
st.markdown(f"As an investment professional, I need to quickly grasp the market sentiment surrounding our portfolio holdings. This section provides AI-generated sentiment scores, helping me identify positive or negative market perceptions that could impact performance.")
st.markdown(f"")
st.subheader("Current News Sentiment Overview")
st.markdown(f"")
st.markdown(f"This table displays the latest AI-generated sentiment scores and signals for our key portfolio holdings, derived from natural language processing (NLP) models like FinBERT analyzing financial news.")

# Display Aggregated News Sentiment Scores in a tabular format (st.dataframe)
# Apply conditional styling within the table
def highlight_sentiment(s):
    if s['sentiment_score'] > 0.3:
        return ['background-color: #C6EFCE'] * len(s) # Light Green
    elif s['sentiment_score'] < -0.3:
        return ['background-color: #FFC7CE'] * len(s) # Light Red
    else:
        return ['background-color: #FFEB9C'] * len(s) # Light Yellow
    
# Display the current sentiment dataframe
display_sentiment_df_st = st.session_state.current_sentiment_df[['ticker', 'sentiment_score', 'n_headlines', 'signal']].copy()
st.dataframe(display_sentiment_df_st.style.apply(highlight_sentiment, axis=1), use_container_width=True)

st.markdown(r"$$ S_{score} > 0.3 \Rightarrow \text{Positive Sentiment (Green)} $$")
st.markdown(r"where $S_{score}$ is the sentiment score, indicating a strong positive market perception.")
st.markdown(r"$$ S_{score} < -0.3 \Rightarrow \text{Negative Sentiment (Red)} $$")
st.markdown(r"where $S_{score}$ is the sentiment score, indicating a strong negative market perception.")
st.markdown(r"$$ -0.3 \le S_{score} \le 0.3 \Rightarrow \text{Neutral Sentiment (Yellow)} $$")
st.markdown(r"where $S_{score}$ is the sentiment score, indicating a balanced or indeterminate market perception.")
st.markdown(f"")

st.subheader("Sentiment Trend Over Time")
st.markdown(f"")
st.markdown(f"Observing the trend of sentiment scores helps me understand if a recent shift is an anomaly or part of a developing pattern. A sharp decline over several days could signal emerging risks.")

# Interactive chart: A bar chart visualizing sentiment by company
# Also, a sentiment time series for selected companies
fig_bar = px.bar(st.session_state.current_sentiment_df, x='ticker', y='sentiment_score', 
                 title='Current Sentiment by Company',
                 color='sentiment_score',
                 color_continuous_scale=['red', 'gray', 'green'],
                 labels={'sentiment_score': 'Sentiment Score', 'ticker': 'Company Ticker'})
st.plotly_chart(fig_bar, use_container_width=True)

selected_tickers = st.multiselect("Select tickers to view sentiment time series", 
                                  options=st.session_state.sentiment_time_series_df['ticker'].unique(), 
                                  default=st.session_state.sentiment_time_series_df['ticker'].unique()[:2])

if selected_tickers:
    filtered_df = st.session_state.sentiment_time_series_df[st.session_state.sentiment_time_series_df['ticker'].isin(selected_tickers)]
    fig_time_series = px.line(filtered_df, x='date', y='sentiment_score', color='ticker', 
                              title='Sentiment Score Trend Over Time',
                              labels={'sentiment_score': 'Sentiment Score', 'date': 'Date', 'ticker': 'Company Ticker'})
    fig_time_series.update_xaxes(rangeselector_buttons=list([
        dict(count=1, label="1m", step="month", stepmode="backward"),
        dict(count=6, label="6m", step="month", stepmode="backward"),
        dict(count=1, label="YTD", step="year", stepmode="todate"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(step="all")
    ]))
    st.plotly_chart(fig_time_series, use_container_width=True)
else:
    st.info("Select one or more tickers to view their sentiment trends.")
```

**Function Calls:** Dataframes (`current_sentiment_df`, `sentiment_time_series_df`) are read from `st.session_state`. Plotly functions (`px.bar`, `px.line`) are used for visualizations.

---

### Page: Credit Risk Analysis

**Markdown:**

```python
st.header("Credit Risk Analysis")
st.markdown(f"")
st.markdown(f"Understanding the credit health of obligors is paramount for managing portfolio risk. This section presents AI-derived credit default probabilities (PDs) and translates them into intuitive implied credit ratings, allowing for rapid risk identification.")
st.markdown(f"")
st.subheader("AI-Derived Credit Default Probabilities")
st.markdown(f"")
st.markdown(f"The table below shows the Probability of Default (PD) for various obligors, along with their AI-implied credit rating and a model agreement score, indicating the model's confidence.")

# Display credit default probabilities in a tabular format (st.dataframe)
# Apply conditional styling to color-code implied ratings
def highlight_rating(s):
    rating = s['rating_implied']
    if rating in ['AAA', 'AA']:
        return ['background-color: #C6EFCE'] * len(s) # Light Green - Very Low Risk
    elif rating in ['A', 'BBB']:
        return ['background-color: #FFFFCC'] * len(s) # Light Yellow - Low/Medium Risk
    elif rating in ['BB', 'B']:
        return ['background-color: #FFD966'] * len(s) # Amber - Elevated Risk
    else: # Default for other ratings (e.g., CCC, D, etc.)
        return ['background-color: #FFC7CE'] * len(s) # Light Red - High Risk

# Ensure 'rating_implied' is treated as a string for consistent styling
display_credit_df_st = st.session_state.credit_df.copy()
display_credit_df_st['rating_implied'] = display_credit_df_st['rating_implied'].astype(str)

st.dataframe(display_credit_df_st.style.apply(highlight_rating, axis=1), use_container_width=True)

st.markdown(r"$$ PD < 0.08 \Rightarrow \text{High Credit Quality (Green)} $$")
st.markdown(r"where $PD$ is the Probability of Default, suggesting a low risk of default.")
st.markdown(r"$$ 0.08 \le PD < 0.25 \Rightarrow \text{Moderate Credit Quality (Yellow)} $$")
st.markdown(r"where $PD$ is the Probability of Default, suggesting a moderate risk of default.")
st.markdown(r"$$ PD \ge 0.25 \Rightarrow \text{Lower Credit Quality (Amber/Red)} $$")
st.markdown(r"where $PD$ is the Probability of Default, suggesting an elevated or high risk of default.")
st.markdown(f"")

st.subheader("Credit Risk Distribution")
st.markdown(f"")
st.markdown(f"A visual representation of Probability of Default (PD) helps in quickly identifying obligors with higher risk.")

# Interactive chart: Heatmap for credit risk distribution
fig_heatmap = px.imshow(
    st.session_state.credit_df[['pd_ensemble']].T, 
    x=st.session_state.credit_df['obligor'], 
    y=['Probability of Default'],
    color_continuous_scale='Hot',
    labels={'x':'Obligor', 'y':'Metric', 'color':'PD Ensemble'},
    title='Credit Default Probability by Obligor'
)
st.plotly_chart(fig_heatmap, use_container_width=True)
```

**Function Calls:** Dataframe (`credit_df`) is read from `st.session_state`. Plotly function (`px.imshow`) is used for visualization.

---

### Page: ESG Governance Report

**Markdown:**

```python
st.header("ESG Governance Report")
st.markdown(f"")
st.markdown(f"Environmental, Social, and Governance (ESG) factors are increasingly important in investment decisions. This section provides a detailed look at governance practices for our portfolio companies, with insights extracted using Retrieval Augmented Generation (RAG) techniques from regulatory filings.")
st.markdown(f"")
st.subheader("Key Governance Metrics")
st.markdown(f"")
st.markdown(f"This report highlights critical governance metrics, especially focusing on the linkage of ESG goals to executive compensation, board structure, and shareholder approval rates. Conditional formatting immediately flags companies that integrate ESG into their compensation structures.")

# Display ESG governance data in a tabular format (st.dataframe)
def highlight_esg_comp(s):
    if str(s['esg_in_compensation']).lower() == 'yes':
        return ['background-color: #C6EFCE'] * len(s) # Light Green for 'Yes'
    elif str(s['esg_in_compensation']).lower() == 'no':
        return ['background-color: #FFC7CE'] * len(s) # Light Red for 'No'
    return [''] * len(s)

st.dataframe(st.session_state.esg_governance_df.style.apply(highlight_esg_comp, axis=1), use_container_width=True)

st.markdown(f"")
st.subheader("ESG Score Overview (Radar Chart)")
st.markdown(f"")
st.markdown(f"The radar chart visually compares the Environmental, Social, and Governance performance for selected companies, helping to identify relative strengths or weaknesses across these pillars.")

# Interactive chart: ESG Radar Chart
if st.session_state.esg_overall_df is not None and not st.session_state.esg_overall_df.empty:
    radar_tickers = st.multiselect("Select companies for ESG Radar Chart", 
                                   options=st.session_state.esg_overall_df['ticker'].unique(), 
                                   default=st.session_state.esg_overall_df['ticker'].unique()[:min(3, len(st.session_state.esg_overall_df))])
    
    if radar_tickers:
        fig_radar = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]])
        for ticker in radar_tickers:
            row = st.session_state.esg_overall_df[st.session_state.esg_overall_df['ticker'] == ticker].iloc[0]
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[row['environmental_score'], row['social_score'], row['governance_score'], row['environmental_score']],
                    theta=['Environmental', 'Social', 'Governance', 'Environmental'],
                    fill='toself',
                    opacity=0.6,
                    name=ticker
                )
            )
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            title_text="ESG Scores by Pillar",
            showlegend=True
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Select one or more companies to view their ESG scores on the radar chart.")
else:
    st.warning("No ESG overall score data available for the radar chart.")
```

**Function Calls:** Dataframes (`esg_governance_df`, `esg_overall_df`) are read from `st.session_state`. Plotly functions (`make_subplots`, `go.Scatterpolar`) are used for visualizations.

---

### Page: Report Generation

**Markdown:**

```python
st.header("Generate AI-Enhanced Reports")
st.markdown(f"")
st.markdown(f"This section allows me to generate various AI-enhanced reports, bringing AI insights into familiar formats like Excel and interactive HTML. This is critical for internal presentations, client communications, and regulatory compliance, directly solving the 'last mile problem' of AI adoption.")
st.markdown(f"")
st.info("Generated reports will be saved to an 'output_reports' directory and made available for download.")

st.subheader("1. AI-Enhanced Portfolio Dashboard (Excel)")
st.markdown(f"")
st.markdown(f"Generates a multi-sheet Excel workbook containing AI-generated sentiment and credit risk data, complete with professional formatting, conditional highlighting, and embedded charts. This report is ideal for detailed daily portfolio monitoring.")
if st.button("Generate Excel Dashboard"):
    with st.spinner("Generating Excel dashboard..."):
        output_file_name = f"ai_portfolio_dashboard_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
        st.session_state.excel_dashboard_path = create_ai_dashboard_excel(
            st.session_state.current_sentiment_df, 
            st.session_state.credit_df, 
            output_file_name
        )
        st.success(f"Excel Dashboard generated at: {st.session_state.excel_dashboard_path}")
if st.session_state.excel_dashboard_path and os.path.exists(st.session_state.excel_dashboard_path):
    with open(st.session_state.excel_dashboard_path, "rb") as file:
        btn = st.download_button(
            label="Download Excel Dashboard",
            data=file,
            file_name=os.path.basename(st.session_state.excel_dashboard_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
st.markdown("---")

st.subheader("2. ESG Governance Report (Excel)")
st.markdown(f"")
st.markdown(f"Creates a focused Excel report on ESG governance, highlighting aspects like ESG linkage in CEO compensation with conditional formatting. This report is tailored for presentations to investment committees or detailed client reviews.")
if st.button("Generate ESG Governance Report"):
    with st.spinner("Generating ESG report..."):
        output_file_name = f"esg_governance_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
        st.session_state.esg_report_path = create_esg_governance_report(
            st.session_state.esg_governance_df, 
            output_file_name
        )
        st.success(f"ESG Governance Report generated at: {st.session_state.esg_report_path}")
if st.session_state.esg_report_path and os.path.exists(st.session_state.esg_report_path):
    with open(st.session_state.esg_report_path, "rb") as file:
        btn = st.download_button(
            label="Download ESG Governance Report",
            data=file,
            file_name=os.path.basename(st.session_state.esg_report_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
st.markdown("---")

st.subheader("3. Interactive HTML Dashboard (Plotly)")
st.markdown(f"")
st.markdown(f"Generates a multi-panel interactive HTML dashboard combining sentiment, ESG, and credit risk visualizations. This provides a dynamic, high-level overview of portfolio health, shareable via any web browser without needing Python installed.")
if st.button("Generate Interactive HTML Dashboard"):
    with st.spinner("Generating interactive dashboard..."):
        output_file_name = f"ai_interactive_dashboard_{datetime.now().strftime('%Y%m%d%H%M%S')}.html"
        st.session_state.html_dashboard_path = create_interactive_dashboard(
            st.session_state.current_sentiment_df, 
            st.session_state.sentiment_time_series_df, 
            st.session_state.esg_overall_df, 
            st.session_state.credit_df, 
            output_file_name
        )
        st.success(f"Interactive HTML Dashboard generated at: {st.session_state.html_dashboard_path}")
if st.session_state.html_dashboard_path and os.path.exists(st.session_state.html_dashboard_path):
    with open(st.session_state.html_dashboard_path, "rb") as file:
        btn = st.download_button(
            label="Download Interactive HTML Dashboard",
            data=file,
            file_name=os.path.basename(st.session_state.html_dashboard_path),
            mime="text/html"
        )
st.markdown("---")

st.subheader("4. Templated HTML Sentiment Report")
st.markdown(f"")
st.markdown(f"Generates a clean, template-based HTML report for current sentiment, ensuring consistency and including all necessary compliance metadata. Ideal for structured internal communications or simplified client updates.")
if st.button("Generate Templated HTML Report"):
    with st.spinner("Generating templated HTML report..."):
        output_file_name = f"sentiment_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.html"
        st.session_state.html_report_path = generate_html_report_from_template(
            st.session_state.current_sentiment_df[['ticker', 'sentiment_score', 'n_headlines', 'signal']], 
            'AI-Enhanced Portfolio Sentiment Summary',
            'Current Portfolio Sentiment Overview',
            score_col_idx=1,
            output_path=output_file_name
        )
        st.success(f"Templated HTML Report generated at: {st.session_state.html_report_path}")
if st.session_state.html_report_path and os.path.exists(st.session_state.html_report_path):
    with open(st.session_state.html_report_path, "rb") as file:
        btn = st.download_button(
            label="Download Templated HTML Report",
            data=file,
            file_name=os.path.basename(st.session_state.html_report_path),
            mime="text/html"
        )
```

**Function Calls:**
*   `st.button("Generate Excel Dashboard")` calls `create_ai_dashboard_excel(st.session_state.current_sentiment_df, st.session_state.credit_df, output_file_name)`.
*   `st.button("Generate ESG Governance Report")` calls `create_esg_governance_report(st.session_state.esg_governance_df, output_file_name)`.
*   `st.button("Generate Interactive HTML Dashboard")` calls `create_interactive_dashboard(st.session_state.current_sentiment_df, st.session_state.sentiment_time_series_df, st.session_state.esg_overall_df, st.session_state.credit_df, output_file_name)`.
*   `st.button("Generate Templated HTML Report")` calls `generate_html_report_from_template(st.session_state.current_sentiment_df[['ticker', 'sentiment_score', 'n_headlines', 'signal']], 'AI-Enhanced Portfolio Sentiment Summary', 'Current Portfolio Sentiment Overview', score_col_idx=1, output_path=output_file_name)`.
*   All generated file paths are stored in `st.session_state` and used to provide `st.download_button` functionality.

---

### Page: Compliance & Model Info

**Markdown:**

```python
st.header("Compliance & AI Model Information")
st.markdown(f"")
st.markdown(f"As a CFA Charterholder, transparency and compliance are paramount when utilizing AI in investment processes. This section provides crucial metadata about the AI models, data sources, and compliance flags associated with the insights presented in this dashboard. This directly addresses **CFA Standard V(B) – Communication**, ensuring ethical and responsible disclosure of AI-assisted content.")
st.markdown(f"")
st.subheader("AI System Metadata")

metadata = {
    'Report Type': 'AI Model Output Dashboard',
    'Generation Date': datetime.now().isoformat(),
    'Models Used': 'FinBERT (sentiment), Stacking Ensemble (credit), RAG (ESG governance)',
    'Data Sources': 'Financial News Feeds (e.g., Yahoo Finance), LendingClub/Credit Data Providers, DEF 14A proxy statements (for RAG)',
    'AI-Generated Content': 'YES - All scores and extractions are model-generated',
    'Human Review Required': 'YES - Verify before client distribution or critical decisions',
    'Compliance Note': 'CFA Standard V(B): AI-assisted content flagged',
    'Disclaimer': 'This report contains AI-generated insights and should not be used as sole basis for investment decisions. Consult human experts.'
}

for key, value in metadata.items():
    st.markdown(f"**{key}:** {value}")

st.markdown(f"")
st.warning("⚠️ **Practitioner Warning**: Every AI-generated cell must be traceable. Without this metadata, an analyst might inadvertently include AI-generated default probabilities in a client report without disclosure—violating CFA Standard V(B) on communication and potentially regulatory requirements. The 'AI-Generated' flag is not optional; it is the compliance safeguard that makes Excel integration responsible. AI outputs in client-facing reports require the same review as human-written content.")
st.markdown(f"")
st.markdown(f"---")
st.markdown(f"")
st.markdown(f"This application was generated using **QuCreate**, an AI-powered assistant, for **educational purposes only**.")
st.markdown(f"Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.")
```

**Function Calls:** None directly on this page. Metadata is displayed using standard Streamlit markdown.

---
