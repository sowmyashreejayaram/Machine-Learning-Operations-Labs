from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.tasks.generate_data import main as generate_data
from scripts.tasks.preprocess import main as preprocess
from scripts.tasks.analyze_sentiment import main as analyze_sentiment
from scripts.tasks.create_visualizations import main as create_viz

default_args = {
    'owner': 'sowmyashree',
    'start_date': datetime(2024, 2, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'sentiment_analysis_pipeline',
    default_args=default_args,
    description='Sentiment analysis MLOps pipeline',
    schedule_interval=None,
    catchup=False,
    tags=['mlops', 'sentiment-analysis'],
)

task1 = PythonOperator(task_id='generate_data', python_callable=generate_data, dag=dag)
task2 = PythonOperator(task_id='preprocess', python_callable=preprocess, dag=dag)
task3 = PythonOperator(task_id='analyze_sentiment', python_callable=analyze_sentiment, dag=dag)
task4 = PythonOperator(task_id='create_visualizations', python_callable=create_viz, dag=dag)

task1 >> task2 >> task3 >> task4
