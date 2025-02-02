from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import requests
import statsmodels
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



# Define Python Functions
def load_data(**kwargs):
    dataset_path = "/opt/airflow/data/Retail_DataSet.csv"
    df = pd.read_csv(dataset_path)
    kwargs['ti'].xcom_push(key='dataframe', value=df)


def external_data(**kwargs):
    df = kwargs['ti'].xcom_pull(key='dataframe', task_ids="load_data")
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], format='%d-%m-%Y', errors='coerce')
    df['PurchaseDate'].fillna(df["PurchaseDate"].mode()[0], inplace=True)
    api_key = "bfaohNRQkv0R9vo3QSQaU1TnbaZwVmWk"
    country_code = 'US'
    start_date = df['PurchaseDate'].min()
    end_date = df['PurchaseDate'].max()
    years = range(start_date.year, end_date.year + 1)
    base_url = "https://calendarific.com/api/v2/holidays"
    holidays_in_range = []
    for year in years:
        params = {
            'api_key': api_key,
            'country': country_code,
            'year': year
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        if 'response' in data and 'holidays' in data['response']:
            for holiday in data['response']['holidays']:
                holiday_date = holiday['date']['iso']
                holiday_name = holiday['name']
                holidays_in_range.append({'date': holiday_date, 'name': holiday_name})
    holiday_dict = {holiday['date']: holiday['name'] for holiday in holidays_in_range}
    df['isHoliday'] = df['PurchaseDate'].dt.strftime('%Y-%m-%d').isin(holiday_dict.keys())
    df['HolidayName'] = df['PurchaseDate'].dt.strftime('%Y-%m-%d').map(holiday_dict).fillna("None")
    kwargs['ti'].xcom_push(key='dataframe_with_external_data', value=df)


def preprocess_data(**kwargs):
    df = kwargs['ti'].xcom_pull(key='dataframe_with_external_data', task_ids='external_data')
    df['Gender'].replace('Unknown', df['Gender'].mode()[0], inplace=True)
    df['Location'].replace('InvalidCity', df['Location'].mode()[0], inplace=True)
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Total_Value'].fillna(df['Total_Value'].mean(), inplace=True)
    invalid_price_rows = df[df['Quantity'] < 0]
    # Count the number of invalid rows
    count_invalid_prices = invalid_price_rows.shape[0]
    # Output the count of invalid rows
    print(f"Number of rows with Quantity less than 0: {count_invalid_prices}")
    df['Quantity'] = df['Quantity'].replace(-1, df['Quantity'].median())
    duplicate_count = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicate_count}")
    df['Age'] = df['Age'].apply(lambda x: round(x) if pd.notnull(x) else x) 
    
    # Identify invalid rows where Total_Value is negative
    invalid_sales_rows = df['Total_Value'] < 0

    # Replace negative Total_Value entries with the mean of valid Total_Value
    df.loc[invalid_sales_rows, 'Total_Value'] = df['Total_Value'][df['Total_Value'] >= 0].mean()
    kwargs['ti'].xcom_push(key='cleaned_dataframe', value=df)


def detect_outliers(**kwargs):
    df = kwargs['ti'].xcom_pull(key='cleaned_dataframe', task_ids='preprocess_data')
    Q1 = df['Total_Value'].quantile(0.25)
    Q3 = df['Total_Value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.2 * IQR
    upper_bound = Q3 + 1.2 * IQR
    df = df[(df['Total_Value'] >= lower_bound) & (df['Total_Value'] <= upper_bound)]
    kwargs['ti'].xcom_push(key='dataframe_without_outliers', value=df)

def get_state_from_city(**kwargs):
    df = kwargs['ti'].xcom_pull(key='dataframe_without_outliers',task_ids='detect_outliers')
    unique_cities = df['Location'].unique()
    city_to_state = {}
    for city in unique_cities:
        print(f"Fetching state for city: {city}")
        url = f"https://nominatim.openstreetmap.org/search?city={city}&format=json"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200 and response.json():
            data = response.json()
            # Extract 'display_name' and split by commas
            display_name = data[0].get("display_name", "")
            parts = display_name.split(", ")
            # Assume state is second-to-last (before country)
            if len(parts) > 2:
                state = parts[-2]
                city_to_state[city]= state
    df['State'] = df['Location'].map(city_to_state)
    kwargs['ti'].xcom_push(key='dataframe_with_state', value=df)

def segment_customer(row):
    if row['High_Spender'] and row['Frequent_Shopper']:
        return 'High Spender & Frequent Shopper'
    elif row['High_Spender']:
        return 'High Spender'
    elif row['Frequent_Shopper']:
        return 'Frequent Shopper'
    else:
        return 'Occasional Shopper'

def KPI(**kwargs):
    df = kwargs['ti'].xcom_pull(key='dataframe_with_state',task_ids='city_from_state')
    customer_group = df.groupby('Customer_ID')
    df['Avg_Spend_Per_Customer'] = df['Customer_ID'].map(customer_group['Total_Value'].mean())
    df['Repeat_Purchases'] = df['Customer_ID'].map(customer_group['Customer_ID'].transform('count'))
    store_group = df.groupby('Store_ID')
    df['Total_Sales_Per_Store'] = df['Store_ID'].map(store_group['Total_Value'].sum())
    category_group = df.groupby('Category')
    df['Avg_Product_Sales_Per_Category'] = df['Category'].map(category_group['Total_Value'].mean())
    spend_threshold = df['Avg_Spend_Per_Customer'].quantile(0.75)
    repeat_threshold = df['Repeat_Purchases'].median()
    df['High_Spender'] = df['Avg_Spend_Per_Customer'] > spend_threshold
    df['Frequent_Shopper'] = df['Repeat_Purchases'] > repeat_threshold
    # Combine labels to identify customer types
    df['Customer_Segment'] = df.apply(segment_customer, axis=1)
    df['YearMonth'] = df['PurchaseDate'].dt.to_period('M')
    df['Age_Group'] = pd.cut(df['Age'],
                                    bins=[0,18, 25, 35, 45, 55, 65, 80],
                                    labels=['0-18','18-25', '26-35', '36-45', '46-55', '56-65', '66-80'])
    # Extract year, month, and day of the week for seasonal analysis
    df['Year'] = df['PurchaseDate'].dt.year
    df['Month'] = df['PurchaseDate'].dt.month
    df['DayOfWeek'] = df['PurchaseDate'].dt.dayofweek
    df.rename(columns={'Location': 'City'}, inplace=True)
    kwargs['ti'].xcom_push(key='data_analysis', value=df)

def model_training(**kwargs):
    df = kwargs['ti'].xcom_pull(key='data_analysis',task_ids='finding_KPIs')
    # Prepare data for clustering: Use Age, Total_Value, and Quantity as features for segmentation
    clustering_data = df[['Age', 'Total_Value', 'Quantity']].dropna()

    # Scale the data for clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clustering_data['Cluster'] = kmeans.fit_predict(scaled_data)

    # Add cluster labels back to the main dataset
    df['Cluster'] = clustering_data['Cluster']
    kwargs['ti'].xcom_push(key='model_training',value=df)

def forcasting_sales(**kwargs):
    df = kwargs['ti'].xcom_pull(key='model_training',task_ids='model_training')
    
    # Prepare data for time-series forecasting
    sales_data = df.groupby('YearMonth')['Total_Value'].sum().reset_index()
    sales_data['YearMonth']
    sales_data.set_index('YearMonth', inplace=True)
    sales_data.index = sales_data.index.to_timestamp()

    # Train-test split
    train_data = sales_data.iloc[:-12]
    test_data = sales_data.iloc[-12:]
    # Fit Exponential Smoothing model
    model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=12).fit()
    # Forecast
    forecast = model.forecast(steps=12)
    # Evaluate performance
    mse = mean_squared_error(test_data, forecast)
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((test_data.values - forecast.values) / test_data.values)) * 100
    accuracy_score = 100 - mape
    combined_data = sales_data.copy()
    combined_data['Type'] = 'Actual'  # Mark all rows as actual initially
    forecast_6_months = model.forecast(steps=6)
    forecast_index = pd.date_range(start=sales_data.index[-1] + pd.DateOffset(months=1), periods=6, freq='M')

    # Append forecasted data (12-month forecast)
    forecast_df = pd.DataFrame({
        'YearMonth': forecast.index,
        'Total_Value': forecast.values,
        'Type': 'Forecast'
    })
    # Append 6-month forecast data
    forecast_6_months_df = pd.DataFrame({
        'YearMonth': forecast_index,
        'Total_Value': forecast_6_months.values,
        'Type': '6-Month Forecast'  
    })
    # Concatenate all data into one DataFrame
    forecast_df = pd.concat([combined_data.reset_index(), forecast_df, forecast_6_months_df])
    kwargs['ti'].xcom_push(key='dataframe_after_forecasting',value=df)
    kwargs['ti'].xcom_push(key='forecast_data', value=forecast_df)




def save_dataframe(**kwargs):
    df = kwargs['ti'].xcom_pull(key='dataframe_after_forecasting', task_ids='forcasting_sales')
    forecast_df = kwargs['ti'].xcom_pull(key='forecast_data',task_ids='forcasting_sales')
    df['YearMonth'] = df['YearMonth'].astype(str)
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or not passed correctly.")
    print("Connecting to sql")
    mysql_conn_string = 'mysql+mysqlconnector://root:root@host.docker.internal/airflow_tableau'
    engine = create_engine(mysql_conn_string)
    df.to_sql('sales_data', con=engine, if_exists='append', index=False)
    forecast_df.to_sql('forecast_data', con=engine, if_exists='append', index=False)


# DAG and Task Definitions
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 27),
    'retries': 1,
}

dag = DAG('retail_data_analysis', default_args=default_args, schedule_interval='@daily')

task_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

task_external_data = PythonOperator(
    task_id='external_data',
    python_callable=external_data,
    dag=dag,
)

task_preprocess_data = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

task_detect_outliers = PythonOperator(
    task_id='detect_outliers',
    python_callable=detect_outliers,
    dag=dag,
)
task_get_state_from_city = PythonOperator(
    task_id = 'city_from_state',
    python_callable=get_state_from_city,
    dag=dag,
)

task_KPI = PythonOperator(
    task_id='finding_KPIs',
    python_callable=KPI,
    dag=dag
)

task_model_training = PythonOperator(
    task_id='model_training',
    python_callable=model_training,
    dag=dag
)

task_forcasting_sales = PythonOperator(
    task_id = 'forcasting_sales',
    python_callable=forcasting_sales,
    dag=dag
)

task_save_dataframe = PythonOperator(
    task_id='save_dataframe',
    python_callable=save_dataframe,
    dag=dag,
)

# Define Task Dependencies
task_load_data >> task_external_data
task_external_data >> task_preprocess_data
task_preprocess_data >> task_detect_outliers
task_detect_outliers >>task_get_state_from_city
task_get_state_from_city >> task_KPI
task_KPI >> task_model_training
task_model_training >> task_forcasting_sales
task_forcasting_sales >> task_save_dataframe
