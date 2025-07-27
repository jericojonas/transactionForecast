# Transaction Forecasting with Machine Learning

This is a **machine learning** based interactive web application to forecast daily transaction volumes using **Facebook Prophet**. The app incorporates Indonesian public holidays and weekends as additional regressors to improve prediction accuracy.

## Features

- Load and visualize historical transaction data
- Incorporate holiday and weekend effects for more accurate forecasting
- Forecast transaction volumes for the next 7 days
- Show percentage changes and status indicators (increase, decrease, stable)
- Detect anomalies where actual transactions deviate significantly from predictions
- Display average transaction statistics for holidays, weekends, and weekdays
- Built with Streamlit for easy deployment and interaction

## Installation

Make sure you have Python 3.7+ installed. Then install the required packages:

```bash
pip install streamlit pandas matplotlib prophet holidays
````

## Usage
Prepare your transaction data in a CSV file named transaksi.csv with at least two columns:

tanggal: date of the transaction (format: YYYY-MM-DD)

jumlah_transaksi: transaction count (numeric, can include thousand separators)

Run the Streamlit app:
````
python -m streamlit run app.py
````

## Data Format Example
| tanggal    | jumlah\_transaksi |
| ---------- | ----------------- |
| 2023-01-01 | 1,000             |
| 2023-01-02 | 1,200             |
| ...        | ...               |


## Technologies Used
Streamlit – interactive web app framework

Pandas – data manipulation

Matplotlib – plotting and visualization

Prophet – forecasting with machine learning

holidays – public holiday data for Indonesia

## License
This project is licensed under the MIT License.
