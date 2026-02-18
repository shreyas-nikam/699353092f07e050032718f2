# QuLab: Lab 36: AI-Enhanced Portfolio Risk & Sentiment Dashboard (Streamlit)

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Table of Contents

- [QuLab: Lab 36: AI-Enhanced Portfolio Risk & Sentiment Dashboard (Streamlit)](#qulab-lab-36-ai-enhanced-portfolio-risk--sentiment-dashboard-streamlit)
  - [Table of Contents](#table-of-contents)
  - [1. Project Overview](#1-project-overview)
  - [2. Features](#2-features)
  - [3. Getting Started](#3-getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [4. Usage](#4-usage)
  - [5. Project Structure](#5-project-structure)
  - [6. Technology Stack](#6-technology-stack)
  - [7. Contributing](#7-contributing)
  - [8. License](#8-license)
  - [9. Contact](#9-contact)
  - [10. Disclaimer](#10-disclaimer)

---

## 1. Project Overview

This Streamlit application, "QuLab: Lab 36: Integrating with Excel/BI Tools," demonstrates an AI-enhanced portfolio risk and sentiment dashboard designed for investment professionals. It addresses the critical "last mile problem" of AI adoption in finance by seamlessly integrating AI-generated insights into familiar tools like Excel dashboards, interactive HTML reports, and this dynamic Streamlit application.

As an investment professional at Apex Capital Management, the daily challenge involves rapidly assimilating vast market data and forward-looking insights—such as market sentiment, credit risk, and ESG factors—into actionable intelligence. Traditionally, this process is manual, time-consuming, and prone to delays. This dashboard automates the integration of sophisticated AI model outputs, ensuring they are immediately accessible and actionable for daily portfolio monitoring and critical decision-making. By streamlining the workflow with proactive AI signals, we can enhance decision-making and maintain robust compliance.

**Key Challenges Addressed:**
*   **Data Overload:** Quickly digesting vast amounts of market and company-specific data.
*   **Actionable Insights:** Translating complex AI model outputs into clear, actionable signals.
*   **Integration Gap:** Bridging the gap between advanced AI analytics and traditional financial reporting tools.
*   **Compliance & Transparency:** Ensuring AI usage adheres to regulatory and ethical standards (e.g., CFA Standard V(B)).

## 2. Features

This application offers a comprehensive suite of tools for portfolio monitoring and risk assessment, driven by AI:

*   **AI-Generated Sentiment Analysis:**
    *   Displays current market sentiment scores and signals for portfolio holdings (e.g., from FinBERT NLP models).
    *   Visualizes sentiment trends over time with interactive line charts (Plotly).
    *   Conditional formatting highlights positive, neutral, and negative sentiment.
*   **Credit Risk Analysis:**
    *   Presents AI-derived credit default probabilities (PDs) for obligors.
    *   Translates PDs into intuitive implied credit ratings with a model agreement score.
    *   Visualizes credit risk distribution using heatmaps (Plotly).
    *   Conditional formatting based on implied credit rating/PD thresholds.
*   **ESG Governance Report (RAG-Enhanced):**
    *   Provides key governance metrics extracted using Retrieval Augmented Generation (RAG) techniques from regulatory filings (e.g., proxy statements).
    *   Highlights aspects like ESG linkage in executive compensation, board structure, and shareholder approval rates.
    *   Interactive radar charts for comparing ESG scores (Environmental, Social, Governance) across selected companies.
    *   Conditional formatting for clear identification of ESG compensation integration.
*   **Automated Report Generation:**
    *   **AI-Enhanced Portfolio Dashboard (Excel):** Generates a multi-sheet Excel workbook with sentiment and credit risk data, professional formatting, conditional highlighting, and embedded charts. Ideal for daily monitoring.
    *   **ESG Governance Report (Excel):** Creates a focused Excel report on ESG governance, tailored for investment committees or client reviews.
    *   **Interactive HTML Dashboard (Plotly):** Produces a multi-panel interactive HTML dashboard combining sentiment, ESG, and credit risk visualizations, shareable via any web browser.
    *   **Templated HTML Sentiment Report:** Generates a clean, template-based HTML report for current sentiment, ensuring consistency and including compliance metadata.
*   **Compliance & AI Model Information:**
    *   Provides crucial metadata about the AI models, data sources, and compliance flags.
    *   Emphasizes the importance of transparency and disclosure (CFA Standard V(B)) for AI-assisted content.
    *   Includes a "Practitioner Warning" on the necessity of traceable and disclosed AI-generated cells.

## 3. Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/qualtlab-36-ai-portfolio.git
    cd qualtlab-36-ai-portfolio
    ```
    *(Note: Replace `your-username/qualtlab-36-ai-portfolio.git` with the actual repository URL)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required Python packages:**
    Create a `requirements.txt` file in the root directory of your project with the following content:
    ```
    streamlit>=1.0.0
    pandas>=1.3.0
    plotly>=5.0.0
    openpyxl>=3.0.0 # For Excel report generation
    Jinja2>=3.0.0 # For HTML report templating
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Ensure `source.py` exists:**
    Verify that a file named `source.py` is present in the same directory as `app.py`. This file contains the helper functions for data generation and report creation.

## 4. Usage

To run the Streamlit application:

1.  **Activate your virtual environment** (if not already active, see [Installation](#installation)).
2.  **Navigate to the project root directory** in your terminal.
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

    This command will open the application in your default web browser (usually at `http://localhost:8000` or `http://localhost:8501`).

4.  **Navigate the Dashboard:**
    *   Use the **sidebar dropdown** to switch between different sections: 'Overview', 'Portfolio Sentiment', 'Credit Risk Analysis', 'ESG Governance Report', 'Report Generation', and 'Compliance & Model Info'.
    *   Interact with charts, select tickers, and trigger report generation using the buttons provided in each section.
    *   Generated Excel and HTML reports will be saved in an `output_reports/` directory created in your project root and become available for download directly within the app.

## 5. Project Structure

```
.
├── app.py                      # Main Streamlit application file
├── source.py                   # Helper functions for data generation, Excel, and HTML report creation
├── requirements.txt            # List of Python dependencies
├── output_reports/             # Directory where generated reports are saved (created upon generation)
└── README.md                   # This README file
```

*(Note: If the templated HTML report uses external templates, a `templates/` directory would be expected, e.g., `templates/report_template.html`)*

## 6. Technology Stack

*   **Python**: The core programming language.
*   **Streamlit**: For building the interactive web dashboard.
*   **Pandas**: For data manipulation and analysis.
*   **Plotly (Express & Graph Objects)**: For creating interactive and publication-quality visualizations (bar charts, line charts, heatmaps, radar charts).
*   **Openpyxl**: For generating and formatting `.xlsx` Excel files.
*   **Jinja2**: (Assumed) For templating HTML reports to ensure consistent structure and branding.
*   **Synthetic Data Generation**: Custom logic within `source.py` to simulate real-world financial data.
*   **AI Models (Conceptual)**:
    *   **FinBERT / NLP Models**: For generating sentiment scores.
    *   **Stacking Ensemble Models**: For deriving credit default probabilities.
    *   **Retrieval Augmented Generation (RAG)**: For extracting structured insights from unstructured text (e.g., ESG governance details from regulatory filings).

## 7. Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add new feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

## 8. License

This project is licensed under the MIT License - see the `LICENSE` file for details.
*(Note: A `LICENSE` file would typically be included in the repository.)*

## 9. Contact

For questions, feedback, or further information, please refer to the QuantUniversity resources:

*   **Website:** [QuantUniversity](https://www.quantuniversity.com/)
*   **Email:** info@quantuniversity.com
*   **LinkedIn:** [QuantUniversity on LinkedIn](https://www.linkedin.com/company/quantuniversity/)

## 10. Disclaimer

This application was generated using **QuCreate**, an AI-powered assistant, for **educational purposes only**.

Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using** for any financial decisions. This dashboard should not be used as the sole basis for investment decisions. Consult human experts and official sources.

## License

## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
