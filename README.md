# BTC Price Prediction App

A deep learning-based web application that predicts Bitcoin (BTC) prices using a Long Short-Term Memory (LSTM) network. The app fetches real-time data from Yahoo Finance and provides an interactive dashboard for visual analysis.

## Features

- **Interactive Charts:** Visualize historical price trends and test-set performance.
- **Deep Learning Model:** Utilizes a pre-trained LSTM model to identify patterns in time-series data.
- **Future Forecasting:** Input a custom number of days to predict future price movements based on the most recent market trends.

## Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **Machine Learning:** Keras, TensorFlow, Scikit-learn
- **Data Analysis:** Pandas, NumPy, Matplotlib, yfinance

## 📂 Project Structure

- `web_bit_coin_price_predictor.py`: The main script to launch the Streamlit web app.
- `bit_coin.ipynb`: Jupyter Notebook containing the initial data analysis and model training logic.
- `Latest_bit_coin_model.keras`: The pre-trained Keras model file.
- `requirements.txt`: Configuration file containing all necessary Python libraries.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [<your-repo-link>](https://github.com/tkuanhoong/bitcoin_price_prediction_model.git)
   cd bitcoin_price_prediction_model
   ```
2. **Set Up a Virtual Environment (Recommended)**

- Create environment
   ```bash
   python -m venv venv
   ```
- Activate (Windows)
  ```bash
  venv\Scripts\activate
  ```

- Activate (Mac/Linux)
  ```bash
  source venv/bin/activate
  ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run & Start the interactive web dashboard**
   ```bash
   streamlit run web_bit_coin_price_predictor.py
   ```
   Once running, the app will be accessible at: http://localhost:8501
