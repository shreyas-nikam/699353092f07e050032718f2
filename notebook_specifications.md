
# AI-Enhanced Portfolio Risk & Sentiment: Delivering Insights to Investment Professionals

As a CFA Charterholder and Investment Professional at **Apex Capital Management**, my daily mandate involves meticulously monitoring our investment portfolios. In today's fast-paced markets, staying ahead requires not just deep financial acumen but also leveraging cutting-edge technology. My biggest challenge is to quickly assimilate vast amounts of market data and forward-looking insights, like market sentiment and credit risk, into actionable intelligence. Traditionally, this meant sifting through countless reports and manually aggregating data, a time-consuming process prone to delays.

This notebook demonstrates a critical workflow: how we integrate AI-generated insights directly into our familiar tools—Excel dashboards and interactive HTML reports. This bridges the "last mile problem" of AI adoption, ensuring that sophisticated AI model outputs are not confined to technical environments but are immediately accessible and actionable for daily portfolio monitoring and critical decision-making. By automating the integration of AI-powered sentiment analysis and credit risk assessments, we can streamline our workflow, enhance our decision-making with proactive signals, and maintain robust compliance.

---

## 1. Setting Up Our Environment and Simulating AI Data Feeds

Before we can integrate AI insights, we need to set up our Python environment and simulate the AI models generating the necessary data. In a real-world scenario, these data would come from trained FinBERT-like models for sentiment, ensemble credit risk models for default probabilities, and RAG extractions for ESG governance. For this lab, we'll generate synthetic datasets that mirror these AI outputs.

### 1.1 Install Required Libraries

We'll install `openpyxl` for Excel file manipulation, `pandas` for data handling, `plotly` for interactive visualizations, and `jinja2` for HTML templating. `schedule` is also included for completeness, as it's crucial for automated reporting pipelines, though we won't execute a scheduled job in this interactive notebook.

```python
!pip install openpyxl pandas plotly jinja2 schedule xlsxwriter
```

### 1.2 Import Dependencies

Next, we import all the necessary Python libraries.

```python
import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta

# For Excel integration
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.chart import BarChart, Reference, LineChart # LineChart for time series
from openpyxl.formatting.rule import CellIsRule
from openpyxl.utils.dataframe import dataframe_to_rows

# For HTML dashboard
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# For HTML templating
from jinja2 import Template

# For logging (though full scheduling isn't run, good practice)
import logging
import os # For managing output paths

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
```

### 1.3 Simulate AI Model Outputs

As an investment professional, I rely on my firm's data science team to provide AI-generated insights. Here, we'll create synthetic datasets that represent these outputs for sentiment, credit risk, and ESG governance. This allows us to focus on the integration and reporting aspects.

```python
def generate_synthetic_data(num_companies=5, num_obligors=4, num_days=10):
    """
    Generates synthetic datasets for sentiment, credit risk, and ESG governance.
    """
    np.random.seed(42)
    companies = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA'][:num_companies]
    obligors = ['Company A', 'Company B', 'Company C', 'Company D'][:num_obligors]
    today = datetime.now().date()

    # --- Sentiment Data ---
    sentiment_data = []
    for ticker in companies:
        for i in range(num_days):
            date = today - timedelta(days=num_days - 1 - i)
            sentiment_score = np.random.uniform(-0.8, 0.8)
            n_headlines = np.random.randint(5, 50)
            signal = 'Neutral'
            if sentiment_score > 0.3:
                signal = 'Positive'
            elif sentiment_score < -0.3:
                signal = 'Negative'
            sentiment_data.append({
                'ticker': ticker,
                'date': date,
                'sentiment_score': sentiment_score,
                'n_headlines': n_headlines,
                'signal': signal
            })
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # Ensure current day's sentiment is available for the dashboard
    current_sentiment_df = sentiment_df[sentiment_df['date'] == today].copy()
    if current_sentiment_df.empty:
        # If today's date not in generated, take the latest for each ticker
        current_sentiment_df = sentiment_df.loc[sentiment_df.groupby('ticker')['date'].idxmax()].copy()
    
    # --- Credit Data ---
    credit_data = []
    for obligor in obligors:
        pd_ensemble = np.random.uniform(0.01, 0.40) # Probability of Default
        rating_implied = ''
        if pd_ensemble < 0.03: rating_implied = 'AAA'
        elif pd_ensemble < 0.08: rating_implied = 'AA'
        elif pd_ensemble < 0.15: rating_implied = 'A'
        elif pd_ensemble < 0.25: rating_implied = 'BBB'
        elif pd_ensemble < 0.35: rating_implied = 'BB'
        else: rating_implied = 'B'

        model_agreement = f"{np.random.randint(1, 4)}/3" # e.g., 3/3 for high agreement
        
        credit_data.append({
            'obligor': obligor,
            'pd_ensemble': pd_ensemble,
            'rating_implied': rating_implied,
            'model_agreement': model_agreement
        })
    credit_df = pd.DataFrame(credit_data)

    # --- ESG Governance Data ---
    esg_governance_data = []
    for ticker in companies:
        ceo_comp = np.random.randint(5, 50) * 100000 # in USD
        board_size = np.random.randint(5, 15)
        esg_in_comp = np.random.choice(['Yes', 'No'], p=[0.7, 0.3]) # ESG metrics in CEO compensation
        esg_governance_data.append({
            'ticker': ticker,
            'ceo_total_compensation': ceo_comp,
            'board_size': board_size,
            'esg_in_compensation': esg_in_comp,
            'independent_directors': np.random.randint(3, board_size),
            'board_independence_pct': np.random.uniform(0.3, 0.9),
            'say_on_pay_approval': np.random.uniform(0.7, 0.99),
            'clawback_policy': np.random.choice(['Yes', 'No'], p=[0.8, 0.2])
        })
    esg_governance_df = pd.DataFrame(esg_governance_data)

    # For Plotly Radar Chart, we need aggregated ESG scores
    esg_scores = []
    for ticker in companies:
        esg_scores.append({
            'ticker': ticker,
            'environmental_score': np.random.uniform(30, 90),
            'social_score': np.random.uniform(30, 90),
            'governance_score': np.random.uniform(30, 90)
        })
    esg_overall_df = pd.DataFrame(esg_scores)
    
    return current_sentiment_df, sentiment_df, credit_df, esg_governance_df, esg_overall_df

# Execute data generation
current_sentiment_df, sentiment_time_series_df, credit_df, esg_governance_df, esg_overall_df = generate_synthetic_data()

# Display a sample of the generated data
print("--- Current Sentiment Data (sample) ---")
print(current_sentiment_df.head())
print("\n--- Credit Risk Data (sample) ---")
print(credit_df.head())
print("\n--- ESG Governance Data (sample) ---")
print(esg_governance_df.head())
print("\n--- ESG Overall Scores (sample for radar chart) ---")
print(esg_overall_df.head())
```

### 1.4 Explanation of Data Generation

As an investment professional, understanding the source and nature of AI-generated data is crucial. This step simulates receiving structured outputs from various AI models:
*   **Sentiment Scores:** Derived from natural language processing (NLP) models (like FinBERT) analyzing financial news, providing a numerical score and a categorical signal (Positive, Neutral, Negative). This helps me gauge market sentiment for our holdings quickly.
*   **Credit Default Probabilities:** Output from sophisticated ensemble credit models, providing a quantitative probability of default (PD) for corporate obligors. This is directly mapped to an **implied credit rating**, a more intuitive measure for risk assessment. The `model_agreement` score gives an indication of the model's confidence or consensus across its sub-components, which is vital for assessing model reliability.
*   **ESG Governance Data:** Extracted from regulatory filings (e.g., DEF 14A proxy statements) using Retrieval Augmented Generation (RAG) techniques, detailing aspects like CEO compensation tied to ESG metrics. This allows for a deeper dive into non-financial risks and opportunities.
*   **Overall ESG Scores:** Simplified scores across Environmental, Social, and Governance pillars, useful for high-level comparison and visualization.

These datasets form the foundation for our reports, allowing us to move from raw AI outputs to structured, decision-ready information.

---

## 2. Generating the AI-Enhanced Portfolio Risk & Sentiment Excel Dashboard

My primary tool for daily portfolio monitoring is Excel. This section focuses on dynamically generating a multi-sheet Excel workbook populated with AI-generated sentiment and credit risk data, complete with professional formatting, conditional highlighting, and embedded charts. This is where AI insights truly become "Excel-native."

### 2.1 Implementing the Excel Dashboard Generator

We'll create a Python function that takes our sentiment and credit data, formats it into an Excel workbook, and adds visualizations and compliance metadata. This function demonstrates how `openpyxl` is used to create a rich, interactive Excel experience without manual intervention.

```python
def create_ai_dashboard_excel(sentiment_df_current, credit_df, output_path='ai_portfolio_dashboard.xlsx'):
    """
    Creates a formatted Excel workbook with AI model outputs for sentiment and credit risk,
    including conditional formatting, charts, and compliance metadata.
    """
    wb = openpyxl.Workbook()

    # --- Styles for professional formatting ---
    header_font = Font(name='Calibri', bold=True, size=12, color='FFFFFF')
    header_fill = PatternFill(start_color='008080', fill_type='solid') # Dark blue
    red_fill = PatternFill(start_color='FFC7CE', fill_type='solid') # Light Red
    green_fill = PatternFill(start_color='C6EFCE', fill_type='solid') # Light Green
    yellow_fill = PatternFill(start_color='FFEB9C', fill_type='solid') # Light Yellow (for neutral sentiment)
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'),
                         top=Side(style='thin'), bottom=Side(style='thin'))

    # --- Sheet 1: Portfolio Sentiment ---
    ws1 = wb.active
    ws1.title = "Portfolio Sentiment"

    # Title and Metadata Header
    ws1['A1'] = 'Daily AI-Enhanced Portfolio Sentiment Dashboard'
    ws1['A1'].font = Font(bold=True, size=16, color='008080') # Dark blue title
    ws1['A2'] = f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Model: FinBERT | AI-Generated'
    ws1['A2'].font = Font(italic=True, size=10, color='666666')

    # Write data headers and apply styling
    # sentiment_df_current should only have one row per ticker for the "current" view.
    # We remove the 'date' column for this aggregated view.
    display_sentiment_df = sentiment_df_current[['ticker', 'sentiment_score', 'n_headlines', 'signal']].copy()
    
    rows = dataframe_to_rows(display_sentiment_df, index=False, header=True)
    for r_idx, row in enumerate(rows, 4): # Start writing data from row 4
        for c_idx, value in enumerate(row, 1):
            cell = ws1.cell(row=r_idx, column=c_idx, value=value)
            cell.border = thin_border
            if r_idx == 4: # Header row
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            else:
                cell.alignment = Alignment(horizontal='center', vertical='center')
    
    # Adjust column widths
    for col_idx, col_name in enumerate(display_sentiment_df.columns, 1):
        ws1.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = max(len(str(col_name)), display_sentiment_df[col_name].astype(str).map(len).max()) + 2

    # Conditional Formatting for sentiment_score
    # Identify the column containing 'sentiment_score'
    score_col_letter = ''
    for col_idx, col_name in enumerate(display_sentiment_df.columns, 1):
        if col_name == 'sentiment_score':
            score_col_letter = openpyxl.utils.get_column_letter(col_idx)
            break
            
    last_row_ws1 = ws1.max_row
    if score_col_letter:
        # Green for Positive: score > 0.3
        ws1.conditional_formatting.add(
            f'{score_col_letter}5:{score_col_letter}{last_row_ws1}',
            CellIsRule(operator='greaterThan', formula=['0.3'], fill=green_fill)
        )
        # Red for Negative: score < -0.3
        ws1.conditional_formatting.add(
            f'{score_col_letter}5:{score_col_letter}{last_row_ws1}',
            CellIsRule(operator='lessThan', formula=['-0.3'], fill=red_fill)
        )
        # Yellow for Neutral: -0.3 <= score <= 0.3
        ws1.conditional_formatting.add(
            f'{score_col_letter}5:{score_col_letter}{last_row_ws1}',
            CellIsRule(operator='between', formula=['-0.3', '0.3'], fill=yellow_fill)
        )

    # Add sentiment bar chart
    chart1 = BarChart()
    chart1.title = "Sentiment by Company"
    chart1.y_axis.title = "Sentiment Score"
    chart1.x_axis.title = "Company Ticker"
    
    data_ref = Reference(ws1, min_col=display_sentiment_df.columns.get_loc('sentiment_score')+1,
                         min_row=5, max_row=last_row_ws1, max_col=display_sentiment_df.columns.get_loc('sentiment_score')+1)
    cats_ref = Reference(ws1, min_col=1, min_row=5, max_row=last_row_ws1)
    
    chart1.add_data(data_ref, titles_from_data=False)
    chart1.set_categories(cats_ref)
    chart1.width = 20
    chart1.height = 12
    
    ws1.add_chart(chart1, f"A{last_row_ws1 + 3}") # Place chart below data

    # --- Sheet 2: Credit Risk ---
    ws2 = wb.create_sheet("Credit Risk")

    # Title and Metadata Header
    ws2['A1'] = 'AI-Derived Credit Default Probabilities'
    ws2['A1'].font = Font(bold=True, size=16, color='008080')
    ws2['A2'] = f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Model: Stacking Ensemble | AI-Generated'
    ws2['A2'].font = Font(italic=True, size=10, color='666666')

    # Write data
    display_credit_df = credit_df[['obligor', 'pd_ensemble', 'rating_implied', 'model_agreement']].copy()

    rows = dataframe_to_rows(display_credit_df, index=False, header=True)
    for r_idx, row in enumerate(rows, 4):
        for c_idx, value in enumerate(row, 1):
            cell = ws2.cell(row=r_idx, column=c_idx, value=value)
            cell.border = thin_border
            if r_idx == 4: # Header row
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            else:
                cell.alignment = Alignment(horizontal='center', vertical='center')

    # Adjust column widths
    for col_idx, col_name in enumerate(display_credit_df.columns, 1):
        ws2.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = max(len(str(col_name)), display_credit_df[col_name].astype(str).map(len).max()) + 2

    # Conditional Formatting for implied ratings
    # Define color mappings for ratings
    rating_colors = {
        'AAA': 'C6EFCE', # Light Green - Very Low Risk
        'AA': 'CCFFCC',
        'A': 'FFFFCC',  # Light Yellow - Low Risk
        'BBB': 'FFD966', # Amber - Medium Risk
        'BB': 'FFC000', # Orange - Elevated Risk
        'B': 'FF0000',  # Red - High Risk
        'CCC': '990000', # Dark Red - Very High Risk (though not in synthetic data, good to have)
        'CC': '660000',
        'C': '330000',
        'D': '000000'
    }
    
    rating_col_letter = ''
    for col_idx, col_name in enumerate(display_credit_df.columns, 1):
        if col_name == 'rating_implied':
            rating_col_letter = openpyxl.utils.get_column_letter(col_idx)
            break
            
    last_row_ws2 = ws2.max_row
    if rating_col_letter:
        for rating, hex_color in rating_colors.items():
            fill_color = PatternFill(start_color=hex_color, fill_type='solid')
            # Formula checks for exact match of rating string
            ws2.conditional_formatting.add(
                f'{rating_col_letter}5:{rating_col_letter}{last_row_ws2}',
                CellIsRule(operator='equal', formula=[f'"{rating}"'], fill=fill_color)
            )

    # --- Sheet 3: Metadata & Compliance ---
    ws3 = wb.create_sheet("Metadata & Compliance")

    metadata = [
        ('Report Type', 'AI Model Output Dashboard'),
        ('Generation Date', datetime.now().isoformat()),
        ('Models Used', 'FinBERT (sentiment), Stacking Ensemble (credit)'),
        ('Data Sources', 'Financial News Feeds, LendingClub/Credit Data Providers'),
        ('AI-Generated Content', 'YES - All scores are model-generated'),
        ('Human Review Required', 'YES - Verify before client distribution'),
        ('Compliance Note', 'CFA Standard V(B): AI-assisted content flagged'),
        ('Disclaimer', 'This report contains AI-generated insights and should not be used as sole basis for investment decisions. Consult human experts.')
    ]

    for i, (key, val) in enumerate(metadata, 1):
        ws3.cell(row=i, column=1, value=key).font = Font(bold=True)
        ws3.cell(row=i, column=2, value=val)
        ws3.column_dimensions[openpyxl.utils.get_column_letter(1)].width = max(len(str(m[0])) for m in metadata) + 2
        ws3.column_dimensions[openpyxl.utils.get_column_letter(2)].width = max(len(str(m[1])) for m in metadata) + 2

    # Save the workbook
    output_dir = 'output_reports'
    os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
    full_output_path = os.path.join(output_dir, output_path)
    wb.save(full_output_path)
    logging.info(f"Excel dashboard saved: {full_output_path}")
    return full_output_path

# Execute the Excel dashboard generation
excel_dashboard_path = create_ai_dashboard_excel(current_sentiment_df, credit_df)
```

### 2.2 Explanation of Excel Dashboard Output

As an Investment Professional, this AI-enhanced Excel dashboard is a game-changer.
The `create_ai_dashboard_excel` function has generated a multi-sheet workbook that is immediately actionable:

*   **"Portfolio Sentiment" Sheet:**
    *   I can instantly see the **current sentiment score** for each of my portfolio holdings.
    *   **Conditional formatting** highlights positive sentiment scores ($S > 0.3$) in green and negative scores ($S < -0.3$) in red, drawing my attention to critical shifts without manual scanning. Neutral sentiment ($-0.3 \le S \le 0.3$) is highlighted in yellow. This thresholding logic, which I'd typically perform mentally or with complex Excel rules, is now applied automatically.
    *   An **embedded bar chart** visually summarizes sentiment across companies, allowing for quick comparisons.
*   **"Credit Risk" Sheet:**
    *   The **Probability of Default (PD)**, a complex AI model output, is translated into an easy-to-understand **implied credit rating**.
    *   **Conditional formatting** on the `rating_implied` column uses a color gradient (e.g., green for 'AAA'/'AA', yellow for 'A'/'BBB', red for 'BB'/'B'), providing immediate visual cues on credit health. This mapping from a numerical $PD$ to a categorical rating helps in rapid risk identification.
    *   The `model_agreement` score gives me a quick sense of the confidence behind the implied rating.
*   **"Metadata & Compliance" Sheet:** This is critical for regulatory adherence. It clearly states the AI models used (e.g., FinBERT, Stacking Ensemble), the generation timestamp, data sources, and most importantly, flags the content as "AI-Generated" and requiring "Human Review." This directly addresses **CFA Standard V(B) – Communication**, ensuring transparency and accountability for AI-assisted content before client distribution.

This automated reporting saves me hours, reduces human error, and ensures I'm always looking at the freshest, AI-informed data, all within the comfort of my primary analytical tool.

---

## 3. Creating the ESG Governance Report in Excel

Beyond market sentiment and credit risk, Environmental, Social, and Governance (ESG) factors are increasingly important. My role requires presenting a clear view of governance practices, especially concerning executive compensation, for internal committees or client discussions. This section automates the generation of a dedicated ESG Governance Excel report.

### 3.1 Implementing the ESG Governance Report Generator

We will develop a function to create a focused Excel report on ESG governance, applying specific formatting to highlight key aspects like ESG linkage in CEO compensation.

```python
def create_esg_governance_report(governance_df, output_path='esg_governance_report.xlsx'):
    """
    Creates a governance comparison report from ESG RAG extraction, with professional formatting.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Governance Comparison"

    # --- Styles ---
    header_font = Font(name='Calibri', bold=True, size=12, color='FFFFFF')
    header_fill = PatternFill(start_color='008080', fill_type='solid')
    green_fill_esg = PatternFill(start_color='C6EFCE', fill_type='solid') # Light Green for 'Yes'
    red_fill_esg = PatternFill(start_color='FFC7CE', fill_type='solid') # Light Red for 'No'
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'),
                         top=Side(style='thin'), bottom=Side(style='thin'))

    # Professional header for the report
    ws.merge_cells('A1:H1') # Merge cells for the main title
    ws['A1'] = 'PORTFOLIO GOVERNANCE COMPARISON'
    ws['A1'].font = Font(bold=True, size=18, color='008080')
    ws['A1'].alignment = Alignment(horizontal='center', vertical='center')

    ws.merge_cells('A2:H2') # Merge cells for source/date
    ws['A2'] = (f'Source: RAG extraction from DEF 14A proxy statements | '
                f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | AI-Assisted')
    ws['A2'].font = Font(italic=True, size=10, color='888888')
    ws['A2'].alignment = Alignment(horizontal='center', vertical='center')

    # Define columns to display and their order
    display_cols = [
        'ticker', 'ceo_total_compensation', 'board_size', 'independent_directors',
        'board_independence_pct', 'esg_in_compensation', 'say_on_pay_approval', 'clawback_policy'
    ]
    
    # Write data headers and apply styling
    for c_idx, col_name in enumerate(display_cols, 1):
        cell = ws.cell(row=4, column=c_idx, value=col_name.replace('_', ' ').title())
        cell.font = Font(bold=True, color='FFFFFF', size=10)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = thin_border

    # Write data rows
    for r_idx, (_, row) in enumerate(governance_df.iterrows(), 5):
        for c_idx, col_name in enumerate(display_cols, 1):
            value = row.get(col_name, 'N/A')
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            cell.border = thin_border
            cell.alignment = Alignment(horizontal='center', vertical='center')

            # Highlight 'esg_in_compensation' column
            if col_name == 'esg_in_compensation':
                if str(value).lower() == 'yes':
                    cell.fill = green_fill_esg
                elif str(value).lower() == 'no':
                    cell.fill = red_fill_esg

    # Adjust column widths dynamically
    for col_idx, col_name in enumerate(display_cols, 1):
        max_length = max(len(str(row.get(col_name, 'N/A'))) for _, row in governance_df.iterrows())
        header_length = len(col_name.replace('_', ' ').title())
        ws.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = max(max_length, header_length) + 2

    # Add a portfolio summary row (example)
    summary_row = ws.max_row + 2
    ws.cell(row=summary_row, column=1, value='PORTFOLIO SUMMARY').font = Font(bold=True, size=12)
    
    # Save the workbook
    output_dir = 'output_reports'
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, output_path)
    wb.save(full_output_path)
    logging.info(f"ESG report saved: {full_output_path}")
    return full_output_path

# Execute the ESG report generation
esg_report_path = create_esg_governance_report(esg_governance_df)
```

### 3.2 Explanation of ESG Governance Report Output

This ESG Governance Report is tailored for presentations to our investment committee (IC) or for detailed client reviews. The `create_esg_governance_report` function generates a clean, one-page Excel report that provides instant insights into crucial governance metrics:

*   **Executive Compensation Linkage:** I can quickly see which companies explicitly link ESG metrics to CEO total compensation. The conditional highlighting (green for 'Yes', red for 'No') immediately flags companies for deeper analysis or discussion regarding their commitment to ESG integration. This helps me assess whether a company's stated ESG goals are genuinely integrated into its incentive structures.
*   **Board Structure:** Metrics like `board_size`, `independent_directors`, and `board_independence_pct` provide a quantitative overview of board oversight.
*   **Shareholder Say-on-Pay:** The `say_on_pay_approval` indicates shareholder satisfaction with executive compensation, offering a signal on governance transparency and accountability.

The professional formatting ensures that this report is presentation-ready, saving me the effort of manually formatting data for IC meetings. Combined with the AI-assisted metadata, it ensures that even these more qualitative, RAG-extracted insights are presented responsibly and transparently.

---

## 4. Visualizing Portfolio Insights with an Interactive HTML Dashboard

While Excel is excellent for detailed tables, an interactive HTML dashboard provides a dynamic, high-level overview of portfolio health, sentiment trends, and risk distributions. This section focuses on generating such a dashboard using Plotly, shareable via any web browser.

### 4.1 Implementing the Interactive Dashboard Generator

We'll build a function that uses `plotly` to create a multi-panel interactive HTML dashboard, combining sentiment, ESG, and credit risk visualizations.

```python
def create_interactive_dashboard(sentiment_df_current, sentiment_time_series_df, esg_overall_df, credit_df, output_path='ai_interactive_dashboard.html'):
    """
    Generates an interactive HTML dashboard combining multiple AI model outputs for daily monitoring.
    Panels: Sentiment bar chart, ESG radar chart, Sentiment Time Series, Credit Risk Heatmap.
    """
    # Create subplots: 2 rows, 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Current News Sentiment by Company', 'ESG Scores by Pillar',
                        'Sentiment Trend Over Time', 'Credit Risk Distribution'),
        specs=[
            [{'type': 'bar'}, {'type': 'polar'}],
            [{'type': 'scatter'}, {'type': 'heatmap'}]
        ]
    )

    # Panel 1: Current Sentiment Bar Chart
    colors_sentiment = ['green' if s > 0.3 else 'red' if s < -0.3 else 'gray'
                        for s in sentiment_df_current['sentiment_score']]
    fig.add_trace(
        go.Bar(x=sentiment_df_current['ticker'],
               y=sentiment_df_current['sentiment_score'],
               marker_color=colors_sentiment,
               name='Current Sentiment'),
        row=1, col=1
    )

    # Panel 2: ESG Radar Chart (if esg_overall_df available)
    if esg_overall_df is not None and len(esg_overall_df) > 0:
        # Loop through a few companies for the radar chart
        for _, row in esg_overall_df.head(min(3, len(esg_overall_df))).iterrows():
            fig.add_trace(
                go.Scatterpolar(
                    r=[row.get('environmental_score'), row.get('social_score'), row.get('governance_score'), row.get('environmental_score')], # Close the loop
                    theta=['Environmental', 'Social', 'Governance', 'Environmental'],
                    fill='toself',
                    opacity=0.5,
                    name=row.get('ticker', 'Unknown')
                ),
                row=1, col=2
            )
    
    # Panel 3: Sentiment Time Series Plot
    # Ensure sentiment_time_series_df is sorted by date
    sentiment_time_series_df_sorted = sentiment_time_series_df.sort_values(by=['ticker', 'date'])
    for ticker in sentiment_time_series_df_sorted['ticker'].unique():
        df_ticker = sentiment_time_series_df_sorted[sentiment_time_series_df_sorted['ticker'] == ticker]
        fig.add_trace(
            go.Scatter(x=df_ticker['date'], y=df_ticker['sentiment_score'],
                       mode='lines+markers', name=f'{ticker} Sentiment'),
            row=2, col=1
        )


    # Panel 4: Credit Risk Heatmap
    # For a heatmap, we need a matrix. Let's simplify: obligor vs. model_agreement values
    # Or, obligor vs pd_ensemble for distribution.
    # To create a basic heatmap, let's represent PD values across obligors
    # We can create a dummy category for 'Risk Level' to represent on y-axis for heatmap if needed.
    # For simplicity, let's make it a single row heatmap showing PDs.
    # A more complex heatmap could involve multiple models or time series of PDs.
    
    # Let's create a simplified heatmap of PDs
    # Transpose for obligors on Y axis, a dummy 'PD' category on X axis
    heatmap_data = credit_df[['obligor', 'pd_ensemble']].set_index('obligor').T
    
    fig.add_trace(
        go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=['Probability of Default'], # Single category for simplified view
            colorscale='Hot', # Example colorscale
            colorbar_title='PD Ensemble'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title_text=(f"AI Model Dashboard | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | AI-Generated"),
        height=800,
        showlegend=True,
        template='plotly_white',
        hovermode='closest' # Better hover experience
    )
    
    # Update axes titles
    fig.update_xaxes(title_text="Company Ticker", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment Score", row=2, col=1)
    
    # Save the dashboard as an HTML file
    output_dir = 'output_reports'
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, output_path)
    fig.write_html(full_output_path, include_plotlyjs='cdn')
    logging.info(f"Interactive dashboard saved: {full_output_path}")
    return full_output_path

# Execute the interactive dashboard generation
html_dashboard_path = create_interactive_dashboard(current_sentiment_df, sentiment_time_series_df, esg_overall_df, credit_df)
```

### 4.2 Explanation of Interactive HTML Dashboard Output

As a CFA Charterholder, this interactive HTML dashboard provides a powerful, dynamic view of our portfolio, complementing the detailed Excel reports. The `create_interactive_dashboard` function generates a single HTML file that I can open in any browser, without needing Python installed.

Here's how this dashboard enhances my daily monitoring:

*   **Current Sentiment (Bar Chart):** A quick glance immediately shows which companies have positive or negative sentiment, similar to Excel, but with richer interactivity.
*   **ESG Scores (Radar Chart):** For selected companies, the radar chart visually compares their Environmental, Social, and Governance performance against each other. This helps in identifying relative strengths or weaknesses across these pillars, especially for ESG-mandated portfolios. For example, if a company has a low governance score relative to its peers, it's an immediate flag for further investigation.
*   **Sentiment Trend (Time Series):** This panel displays how sentiment for individual holdings has evolved over time. Observing the trend helps me understand if a recent sentiment shift is an anomaly or part of a developing pattern. A sharp decline in sentiment over several days, for instance, could signal emerging risks.
*   **Credit Risk (Heatmap):** The heatmap provides a visual distribution of Probability of Default (PD) across different obligors. While simplified here, in a real scenario, this could show PDs across various models or time points, allowing for quick identification of high-risk obligors or trends in credit quality across the portfolio. Higher PDs would be depicted in "hotter" colors, drawing immediate attention.

The interactivity (zooming, panning, hovering for details) allows for a more fluid and exploratory analysis than static images, making it an invaluable tool for daily portfolio oversight and quickly briefing stakeholders on key trends.

---

## 5. Generating Templated HTML Reports with Compliance Metadata

For structured internal communications or even simplified client updates, a clean, template-based HTML report ensures consistency and includes all necessary compliance metadata. This section demonstrates generating such a report using Jinja2, separating the data from its presentation.

### 5.1 Implementing the Templated HTML Report Generator

We'll define a Jinja2 template for our HTML report and then use it to render our sentiment data, ensuring the inclusion of all required compliance metadata.

```python
def generate_html_report_from_template(data_df, title, section_title, score_col_idx=1, output_path='sentiment_report.html'):
    """
    Generates an HTML report from a DataFrame using a Jinja2 template, including compliance metadata.
    score_col_idx is 0-indexed for the DataFrame column that represents sentiment_score for conditional formatting.
    """

    REPORT_TEMPLATE = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #333; }
        h1 { color: #008080; border-bottom: 2px solid #008080; padding-bottom: 10px; }
        h2 { color: #008080; margin-top: 30px; }
        .metadata { font-size: 12px; margin-bottom: 20px; color: #666; }
        .ai-flag { background: #FFF3CD; padding: 8px; border-radius: 4px; border-left: 4px solid #FFC107; margin: 10px 0; font-weight: bold; color: #8B4513;}
        table { border-collapse: collapse; width: 100%; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        th { background: #008080; color: white; padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
        td { border: 1px solid #ddd; padding: 10px 15px; text-align: center; }
        tr:nth-child(even) { background: #f9f9f9; }
        .positive { color: green; font-weight: bold; }
        .negative { color: red; font-weight: bold; }
        .neutral { color: gray; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <div class="metadata">
        Generated: {{ timestamp }} | Models: {{ models }} | Report ID: {{ report_id }}
    </div>
    <div class="ai-flag">
        This report contains AI-generated content. All data should be verified before use in client communications.
    </div>

    <h2>{{ section_title }}</h2>
    <table>
        <thead>
            <tr>
                {% for col in columns %}
                    <th>{{ col }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in data %}
                <tr>
                    {% for val in row %}
                        <td{% if loop.index0 == score_col_idx %}
                            class="{{ 'positive' if val > 0.3 else 'negative' if val < -0.3 else 'neutral' }}"
                        {% endif %}>{{ val | round(3) if loop.index0 == score_col_idx else val }}</td>
                    {% endfor %}
                </tr>
            {% endfor %}
        </tbody>
    </table>
    <div class="metadata">
        Compliance: CFA Standard V(B) - AI content flagged | Human review required before distribution.
    </div>
</body>
</html>
    """)

    # Prepare data for rendering
    html_content = REPORT_TEMPLATE.render(
        title=title,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        models='FinBERT',
        report_id=f'RPT-{datetime.now().strftime("%Y%m%d%H%M%S")}',
        section_title=section_title,
        columns=data_df.columns.tolist(),
        data=data_df.values.tolist(),
        score_col_idx=score_col_idx # Pass the index of the sentiment score column
    )
    
    # Save the HTML file
    output_dir = 'output_reports'
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, output_path)
    with open(full_output_path, 'w') as f:
        f.write(html_content)
    logging.info(f"HTML report saved: {full_output_path}")
    return full_output_path

# Execute HTML report generation for current sentiment
# The sentiment_score is at index 1 in current_sentiment_df[['ticker', 'sentiment_score', 'n_headlines', 'signal']]
html_report_path = generate_html_report_from_template(
    current_sentiment_df[['ticker', 'sentiment_score', 'n_headlines', 'signal']], # Select columns for the report
    'AI-Enhanced Portfolio Sentiment Summary',
    'Current Portfolio Sentiment Overview',
    score_col_idx=1 # 'sentiment_score' is the second column (index 1) in the passed DataFrame
)
```

### 5.2 Explanation of Templated HTML Report Output

As a CFA Charterholder, producing consistent and compliant reports is a core responsibility. The `generate_html_report_from_template` function, utilizing `Jinja2`, helps me achieve this by separating data from presentation logic.

Here's why this is valuable:

*   **Consistency and Branding:** The template ensures that every report generated adheres to a standard layout, including our firm's branding elements (implied by the CSS styling), titles, and the crucial AI compliance flags. This consistency is essential for professional communication.
*   **Dynamic Content:** The same template can be used with different datasets, effortlessly populating the report with new AI insights daily or weekly without any manual formatting.
*   **Compliance Baked In:** The template explicitly includes all necessary metadata: the generation timestamp, the AI models used, a clear "AI-Generated" flag, and the mandatory "Human Review Required" warning. This directly addresses **CFA Standard V(B) – Communication**, ensuring that stakeholders are fully aware of the nature of the content. This is a robust control against inadvertent disclosure of unverified AI outputs.
*   **Readability:** For quick internal reviews or non-technical stakeholders, an HTML report is often more accessible than an Excel file, especially when distributed via email or internal portals.

The conditional styling within the HTML template for sentiment scores (e.g., green for positive, red for negative) echoes the visual cues from our Excel dashboard, reinforcing key insights in a web-friendly format. This principle of separating data from its presentation layer ($D \rightarrow P$) is fundamental for scalable and maintainable reporting systems at Apex Capital Management.

---

## Conclusion

This notebook has guided me, an Investment Professional at Apex Capital Management, through the critical workflow of integrating AI-generated financial insights into my daily operations. By transforming raw AI model outputs into professionally formatted Excel dashboards and interactive HTML reports, we've addressed the "last mile problem" of AI adoption.

I've learned to:
*   **Process AI-generated sentiment scores** and display them with intuitive conditional formatting in Excel and on an interactive dashboard.
*   **Integrate and process AI-derived credit default probabilities**, translating complex PDs into actionable implied credit ratings within an Excel report and visualizing risk distributions in an HTML dashboard.
*   **Generate comprehensive compliance metadata** for all AI outputs, ensuring adherence to ethical guidelines like CFA Standard V(B) and maintaining transparency.
*   **Leverage Python libraries** like Pandas, OpenPyXL, Plotly, and Jinja2 for structured data manipulation, robust report generation, and dynamic visualization.
*   **Produce formatted reports** that are not just informative but also ready for review by internal committees and clients.

This hands-on experience has solidified my understanding of how AI can be practically deployed to enhance efficiency, improve decision-making, and ensure regulatory compliance in portfolio management, making AI an indispensable tool in my professional toolkit.
```