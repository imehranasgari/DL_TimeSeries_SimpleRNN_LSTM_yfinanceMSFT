# Time-Series Forecasting of Microsoft Stock Prices using RNN & LSTM

This project demonstrates the implementation of Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models to forecast Microsoft (MSFT) stock prices. The goal is to predict the closing price two days in advance based on the previous six days of data, showcasing a practical application of deep learning for time-series analysis.

This file was developed to demonstrate skills in implementing and explaining machine learning models, rather than solely focusing on achieving the highest evaluation metrics. The use of a SimpleRNN alongside an LSTM is intentional to benchmark and illustrate fundamental concepts in handling sequential data.

## Problem Statement and Goal of Project

The primary objective is to build and evaluate sequential deep learning models capable of predicting future stock market movements. By analyzing historical stock data for Microsoft from 2012 to 2020, the project aims to forecast the closing prices for 2021. This serves as a practical exercise in time-series forecasting, a critical task in financial analysis and algorithmic trading.

## Solution Approach

1.  **Data Acquisition**: Historical stock data for Microsoft ('MSFT') was downloaded using the `yfinance` library, spanning from January 1, 2012, to December 31, 2021.
2.  **Data Preprocessing**:
      * The dataset was split into a training set (2012-2020) and a testing set (2021).
      * The 'Close' price was scaled using `MinMaxScaler` to normalize the data between 0 and 1, which helps improve model training stability.
      * Sequential data structures were created where each input sample (`X`) consists of 6 consecutive days of closing prices, and the corresponding target (`y`) is the closing price for the subsequent 2 days.
3.  **Model Implementation**:
      * A **Simple Recurrent Neural Network (SimpleRNN)** was built as a baseline sequential model.
      * A **Long Short-Term Memory (LSTM)** network was implemented to leverage its superior ability to capture long-term dependencies in data.
      * Both models consist of two recurrent layers followed by a Dense output layer to predict the two-step-ahead forecast.
4.  **Training and Evaluation**: The models were trained on the preprocessed training data for 100 epochs using the Adam optimizer and Mean Squared Error loss function. Their predictive performance was then visualized by plotting the forecasted prices against the actual prices for the test year (2021).

## Technologies & Libraries

  * **Python 3.10**
  * **TensorFlow & Keras**: For building and training the deep learning models.
  * **Scikit-learn**: For data preprocessing, specifically `MinMaxScaler`.
  * **Pandas**: For data manipulation and analysis.
  * **NumPy**: For numerical operations.
  * **yfinance**: For fetching historical stock market data.
  * **Matplotlib**: For data visualization and plotting results.

## Description about Dataset

The dataset consists of daily stock market data for **Microsoft (MSFT)** from January 1, 2012, to December 31, 2021. It includes the following features: Open, High, Low, Close, and Adjusted Close prices, along with trading Volume. For this forecasting task, only the **'Close'** price was used as the feature for model training and prediction.

## Installation & Execution Guide

To replicate this project, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/imehranasgari/your-repo-name.git
    cd your-repo-name
    ```
2.  **Install the required libraries:**
    ```bash
    pip install tensorflow numpy pandas scikit-learn yfinance matplotlib
    ```
3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook SMDL_me.ipynb
    ```

## Key Results / Performance

The models were evaluated by visually comparing their predictions against the actual stock prices for 2021. Both the SimpleRNN and LSTM models successfully captured the overall trend of the MSFT stock price throughout the test year. The LSTM model, as expected, provided a slightly closer fit to the real price fluctuations, demonstrating its effectiveness in handling time-series data. The resulting plots serve as a clear qualitative measure of the models' forecasting capabilities.

## Sample Output

Below are the prediction plots for the SimpleRNN and LSTM models, comparing the actual vs. predicted MSFT stock prices for 2021.

**SimpleRNN Model Prediction vs. Real Price**

**LSTM Model Prediction vs. Real Price**

## Additional Learnings / Reflections

This project was a valuable exercise in applying recurrent neural networks to a real-world financial dataset. Key takeaways include:

  * **Data Preparation is Crucial**: Structuring time-series data into appropriate sequential windows (`X_steps` and `y_steps`) is fundamental for training RNN and LSTM models effectively.
  * **Model Selection**: While both models performed well, the exercise highlights the architectural strengths of LSTMs in capturing dependencies over longer sequences compared to SimpleRNNs.
  * **Practical Application**: The project successfully demonstrates a foundational approach to algorithmic trading and financial forecasting, providing a solid base for developing more complex predictive systems.

💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*

-----

## 👤 Author

**Mehran Asgari**

  * **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com)
  * **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari)

-----

## 📄 License

This project is licensed under the Apache 2.0 License – see the `LICENSE` file for details.