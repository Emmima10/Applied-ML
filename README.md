#  AI-Driven Business Analytics for Superstore Giant

This project leverages machine learning, time-series forecasting, and large language models (LLMs) to deliver interactive and automated business insights for a retail superstore dataset.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `Sales_Forecast.ipynb` | Forecasting sales using ARIMA, Prophet|
| `Superstore_TimeSeries_analysis.ipynb` | Time-series visualizations and trend analysis |
| `Superstore_Daily_analysis.ipynb` | Daily-level data analysis |
| `Customer_Segmentation.ipynb` | Customer clustering based on behavior (sales, profit, discount) |
| `Mixed_Data_Customer_Segmentation.ipynb` | Alternative segmentation using mixed-type clustering |
| `chatbot.py` | Streamlit app for querying business insights via LLM |
| `chatbotkb.txt` | Knowledge base text file with EDA/modeling insights for chatbot |
| `requirements.txt` | Python dependencies |

---

## 🚀 Features

### 📊 1. Exploratory Data Analysis (EDA)
- Sales trends by category, region, and segment
- Discount impact on profit
- Correlation heatmaps and visual insights

### 🔮 2. Sales Forecasting
- **Prophet** (MAPE ≈ 13.9%) – models trends and seasonality
- **ARIMA** – statistical baseline
- **XGBoost** – fine-grained daily prediction (MAPE ≈ 7.9%)

### 👥 3. Customer Segmentation
- Clustering using sales, profit, discount, and quantity
- Business strategy for each customer cluster

### 💬 4. LLM-Powered Chatbot
- Built using `Flan-T5` from Hugging Face
- Responds to natural language questions like:
  - “Which region had the highest sales?”
  - “How did XGBoost perform?”
- Runs as a **Streamlit app**, sourcing answers from a fact-based knowledge file

---

## 📦 Setup Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Run the chatbot
streamlit run chatbot.py
