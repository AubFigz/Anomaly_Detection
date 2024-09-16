import os
import pandas as pd
import numpy as np
import pickle  # For saving/loading trained models
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import layers, models
import mlflow
import mlflow.sklearn
import mlflow.keras
from kafka import KafkaConsumer
import pyspark
from pyspark.sql import SparkSession
import redis
import flask
import threading
import asyncio
import time
import logging
import json
from kubernetes import client, config as k8s_config
import tensorflow.keras.backend as K
from shap import KernelExplainer, summary_plot
from flasgger import Swagger
from prometheus_client import Gauge, start_http_server
import redis
import cv2  # For handling CCTV footage
from rasa.core.agent import Agent  # For chatbot integration
from rasa.utils.endpoints import EndpointConfig  # For AI assistant integration
import boto3  # For AWS integration
from google.cloud import storage  # For Google Cloud integration
from concurrent.futures import ThreadPoolExecutor
from cryptography.fernet import Fernet  # For encryption and security
from yolo import YOLO  # Advanced CCTV Object Detection using YOLO
from keras.models import load_model
import optuna  # For hyperparameter optimization
import yaml  # For loading configuration

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.FileHandler("anomaly_detection.log"), logging.StreamHandler()])

# AWS SNS setup for sending alerts
sns_client = boto3.client('sns', region_name=config['aws']['region'])

# Load YOLO model for CCTV object detection
yolo = YOLO(
    weights_path=config['models']['yolo_weights'],
    config_path=config['models']['yolo_cfg'],
    labels_path=config['models']['yolo_classes']
)


# Function to save models to a file
def save_model(model, filename):
    """Save trained model to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


# Function to load saved models
def load_model(filename):
    """Load model from a file if it exists."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None


# Feature Extraction: Process CCTV Footage using YOLO
def process_cctv_with_yolo(video_path, frame_resize=(416, 416)):
    """
    Process CCTV footage using YOLO for anomaly detection based on object presence.
    Args:
        video_path (str): Path to CCTV footage video file.
        frame_resize (tuple): Size to resize frames for processing speed.

    Returns:
        list: Detected anomalies in the video.
    """
    cap = cv2.VideoCapture(video_path)
    anomalies = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_resize)  # Resize for faster processing
        objects_detected = yolo.detect_objects(frame)
        for obj in objects_detected:
            if obj in ["suspicious_object", "loitering", "unauthorized_person"]:
                anomalies.append(obj)
    cap.release()
    return anomalies


# 1. Data Ingestion and Preprocessing with Asynchronous Streaming and Data Security
def init_spark_session(app_name="RealTimeSecurityAnomalyDetection"):
    """
    Initialize Spark session for distributed processing.
    Args:
        app_name (str): Name of the Spark application.

    Returns:
        SparkSession: Initialized Spark session.
    """
    spark = (SparkSession.builder
             .appName(app_name)
             .config("spark.executor.memory", "4g")
             .config("spark.executor.cores", "4")
             .getOrCreate())
    return spark


async def load_data_from_kafka(topic_name, retries=3):
    """
    Load data from a Kafka topic asynchronously with retry logic.
    Args:
        topic_name (str): Kafka topic name.
        retries (int): Number of retries for connection failure.

    Yields:
        pd.DataFrame: Loaded data from Kafka.
    """
    attempt = 0
    while attempt < retries:
        try:
            consumer = KafkaConsumer(
                topic_name,
                bootstrap_servers=config['kafka']['bootstrap_servers'],
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id=config['kafka']['group_id'],
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            for message in consumer:
                yield pd.DataFrame([message.value])  # Yield data in chunks
        except Exception as e:
            attempt += 1
            logging.error(f"Kafka connection failed: {e}, retrying {attempt}/{retries}")
            time.sleep(5)


def encrypt_data(data, encryption_key):
    """
    Encrypt data using Fernet symmetric encryption.
    Args:
        data (str): Data to encrypt.
        encryption_key (str): Encryption key.

    Returns:
        str: Encrypted data.
    """
    try:
        fernet = Fernet(encryption_key)
        return fernet.encrypt(data.encode())
    except Exception as e:
        logging.error(f"Encryption failed: {e}")
        return None


def decrypt_data(encrypted_data, encryption_key):
    """
    Decrypt data using Fernet symmetric encryption.
    Args:
        encrypted_data (str): Encrypted data to decrypt.
        encryption_key (str): Encryption key.

    Returns:
        str: Decrypted data.
    """
    try:
        fernet = Fernet(encryption_key)
        return fernet.decrypt(encrypted_data).decode()
    except Exception as e:
        logging.error(f"Decryption failed: {e}")
        return None


def preprocess_data(df, scaling_method='standard', pca_var=0.95):
    """
    Preprocess data by filling missing values, scaling, and reducing dimensions.
    Args:
        df (pd.DataFrame): Data to preprocess.
        scaling_method (str): Scaling method ('standard' or 'minmax').
        pca_var (float): Variance to preserve for PCA dimensionality reduction.

    Returns:
        np.array: Preprocessed data.
    """
    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    df.fillna(0, inplace=True)
    normalized_data = scaler.fit_transform(df)

    if pca_var:
        pca = PCA(n_components=pca_var)
        reduced_data = pca.fit_transform(normalized_data)
        return reduced_data
    return normalized_data


def load_encryption_key():
    """
    Load encryption key from environment variables.
    Returns:
        str: Encryption key.
    """
    return config['encryption_key']


def extract_network_features(log_data):
    """
    Extract network features such as packet size and data rate from logs.
    Args:
        log_data (pd.DataFrame): Network log data.

    Returns:
        np.array: Extracted features.
    """
    packet_size = log_data['packet_size'].mean()
    data_rate = log_data['data_rate'].mean()
    return np.array([packet_size, data_rate])


# 2. Model Implementation with AutoML (Optuna)
def optimize_isolation_forest(trial):
    """
    Optimize Isolation Forest hyperparameters using Optuna.
    Args:
        trial: Optuna trial object.

    Returns:
        IsolationForest: Optimized Isolation Forest model.
    """
    contamination = trial.suggest_float("contamination", 0.001, 0.1)
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    model = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
    return model


def optimize_autoencoder(data, trial):
    """
    Optimize Autoencoder architecture using Optuna.
    Args:
        data (np.array): Input data for the Autoencoder.
        trial: Optuna trial object.

    Returns:
        models.Sequential: Optimized Autoencoder model.
    """
    encoding_dim = trial.suggest_int("encoding_dim", 8, 64)
    input_dim = data.shape[1]
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    autoencoder = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(encoding_dim, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(input_dim, activation='sigmoid')
    ])
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder


def run_automl(data):
    """
    Run AutoML using Optuna for hyperparameter tuning and model optimization.
    Args:
        data (np.array): Data for model training.

    Returns:
        optuna.Trial: Best trial with optimized model.
    """
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    def objective(trial):
        mlflow.start_run()
        model = optimize_isolation_forest(trial)
        model.fit(data)
        mlflow.log_params(trial.params)
        mlflow.sklearn.log_model(model, "model")
        mlflow.end_run()
        return model

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    return study.best_trial


# Periodic retraining of models
def periodic_retrain(model_filename, data, retrain_interval=86400):
    """Periodic retraining of the model using updated data."""
    while True:
        logging.info("Periodic retraining started...")
        best_trial = run_automl(data)
        save_model(best_trial.params['model'], model_filename)
        logging.info(f"Model saved as {model_filename}. Next retrain in {retrain_interval} seconds.")
        time.sleep(retrain_interval)


# 3. Real-Time Detection and Alerts with AWS SNS
async def real_time_detection(model_filename, data_stream, detection_type='isolation_forest', batch_size=10):
    """
    Real-time anomaly detection with batching and model prediction.
    Args:
        model_filename (str): Path to the saved model.
        data_stream: Stream of input data for detection.
        detection_type (str): Type of model used ('isolation_forest', etc.).
        batch_size (int): Number of data points to process in a batch.

    Yields:
        bool: Whether an anomaly was detected.
    """
    model = load_model(model_filename)
    if model is None:
        raise Exception(f"Model not found: {model_filename}")

    batch_data = []
    async for data in data_stream:
        batch_data.append(data)
        if len(batch_data) >= batch_size:
            is_anomaly = check_for_anomaly(model, batch_data, detection_type)
            if is_anomaly:
                send_alert(f"Anomaly detected in batch: {batch_data}")
            batch_data = []


def check_for_anomaly(model, batch_data, detection_type):
    """
    Check if an anomaly exists in the batch data based on the model's prediction.
    Args:
        model: Trained model.
        batch_data (list): List of data to check for anomalies.
        detection_type (str): Type of detection model.

    Returns:
        bool: True if an anomaly is detected, False otherwise.
    """
    is_anomaly = False
    if detection_type == 'isolation_forest':
        predictions = model.predict(batch_data)
        is_anomaly = any(pred == -1 for pred in predictions)
    # Add logic for other models if necessary
    return is_anomaly


# 4. AWS SNS for Real-time Alerts
def send_alert(message, retries=3):
    """
    Send an alert via AWS SNS with retry logic.
    Args:
        message (str): Message to send as an alert.
        retries (int): Number of retries if SNS fails.

    Returns:
        None
    """
    attempt = 0
    while attempt < retries:
        try:
            sns_client.publish(
                TopicArn=config['aws']['sns_topic_arn'],
                Message=message,
                Subject="Anomaly Detected in Security System"
            )
            logging.info(f"Alert sent: {message}")
            break
        except Exception as e:
            attempt += 1
            logging.error(f"Failed to send alert: {e}, retrying {attempt}/{retries}")
            time.sleep(3)

    if attempt == retries:
        logging.error(f"Alert failed after {retries} attempts. Message: {message}")


# 5. Edge Computing and Federated Learning for Low-Latency Environments
def deploy_on_aws_lambda(model_path, function_name="AnomalyDetectionFunction"):
    """
    Deploy the anomaly detection model to AWS Lambda for low-latency environments.
    Args:
        model_path (str): Path to the model file.
        function_name (str): AWS Lambda function name.

    Returns:
        None
    """
    client = boto3.client('lambda', region_name=config['aws']['region'])
    with open(model_path, 'rb') as model_file:
        response = client.update_function_code(
            FunctionName=function_name,
            ZipFile=model_file.read()
        )
    logging.info(f"AWS Lambda deployment response: {response}")


def deploy_on_google_cloud(model_path, bucket_name='anomaly-detection-models'):
    """
    Deploy the model to Google Cloud Storage for scalability.
    Args:
        model_path (str): Path to the model file.
        bucket_name (str): Google Cloud Storage bucket name.

    Returns:
        None
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(os.path.basename(model_path))
    blob.upload_from_filename(model_path)
    logging.info(f"Model uploaded to Google Cloud Storage bucket: {bucket_name}")


# 6. Advanced Monitoring: Prometheus and Latency Tracking
def track_model_latency(start_time, data_points, model_name):
    """
    Track and log model latency and number of data points processed.
    Args:
        start_time (float): Start time of the model inference.
        data_points (int): Number of data points processed.
        model_name (str): Name of the model used.

    Returns:
        float: Calculated latency.
    """
    latency = time.time() - start_time
    logging.info(f"{model_name} processed {data_points} data points in {latency:.4f} seconds.")
    return latency


# Start Prometheus HTTP server for monitoring metrics
start_http_server(8000)
anomaly_gauge = Gauge('anomaly_count', 'Number of anomalies detected')
latency_gauge = Gauge('model_latency', 'Latency of model inference')


def update_metrics(anomalies_detected, latency):
    """
    Update Prometheus metrics for anomalies detected and model latency.
    Args:
        anomalies_detected (int): Number of detected anomalies.
        latency (float): Latency of the model inference.

    Returns:
        None
    """
    anomaly_gauge.set(anomalies_detected)
    latency_gauge.set(latency)


# 7. Model Retraining and Auto-Scaling with Cloud Functions
def retrain_model(model, data, detection_type='isolation_forest'):
    """
    Retrain the anomaly detection model with new data.
    Args:
        model: The model to retrain.
        data (np.array): New data for retraining.
        detection_type (str): Type of the model being retrained.

    Returns:
        None
    """
    logging.info(f"Retraining {detection_type} model with new data...")
    mlflow.start_run()
    model.fit(data)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_param("model_type", detection_type)
    mlflow.end_run()


# 8. Flask-based Dashboard Enhancements: Add real-time visualizations using Plotly/Dash
def create_flask_dashboard():
    """
    Create a Flask-based dashboard for monitoring anomalies and metrics.
    Returns:
        None
    """
    app = flask.Flask(__name__)
    Swagger(app)

    @app.route('/anomalies', methods=['GET'])
    def get_anomalies():
        logging.info("Anomalies request received.")
        return flask.jsonify({"status": "No anomalies detected"})

    @app.route('/metrics', methods=['GET'])
    def get_metrics():
        anomalies = anomaly_gauge._value.get()
        latency = latency_gauge._value.get()
        logging.info(f"Metrics request received. Anomalies: {anomalies}, Latency: {latency}")
        return flask.jsonify({"anomalies_detected": anomalies, "latency": latency})

    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000)).start()


# 9. AI Chatbot Enhancement: Integrate with Rasa for anomaly insights
def initialize_rasa_chatbot():
    """
    Initialize Rasa chatbot for querying anomaly insights.
    Returns:
        Agent: Initialized Rasa agent.
    """
    endpoint = EndpointConfig(config['rasa']['endpoint'])
    agent = Agent.load("models/dialogue", action_endpoint=endpoint)
    return agent


def handle_user_query(agent, user_input):
    """
    Handle user query via Rasa chatbot and return response.
    Args:
        agent: Rasa chatbot agent.
        user_input (str): Input query from the user.

    Returns:
        list: Chatbot response.
    """
    response = agent.handle_text(user_input)
    return response


# Main Execution Flow
if __name__ == "__main__":
    # Initialize the Spark session for distributed processing
    spark = init_spark_session()

    # Load encryption key from environment variable
    encryption_key = load_encryption_key()


    # Asynchronously load and process real-time data
    async def run_detection_pipeline():
        async for raw_data in load_data_from_kafka(config['kafka']['topic_name']):
            encrypted_data = encrypt_data(str(raw_data), encryption_key)
            decrypted_data = decrypt_data(encrypted_data, encryption_key)
            data = preprocess_data(pd.DataFrame([json.loads(decrypted_data)]))

            # Save and load the trained model periodically
            model_filename = "saved_model.pkl"
            await real_time_detection(model_filename, data)


    # Start the pipeline asynchronously
    asyncio.run(run_detection_pipeline())

    # Start periodic retraining
    periodic_retrain("saved_model.pkl", pd.DataFrame([json.loads(decrypted_data)]))
