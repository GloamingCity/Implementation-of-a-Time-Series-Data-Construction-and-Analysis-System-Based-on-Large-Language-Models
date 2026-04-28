"""
数据集类型配置文件
根据 Data_Collection.xlsx 中的任务描述自动归类
"""
from pathlib import Path

TYPE_PREDICTION = "prediction"
TYPE_ANOMALY_DETECTION = "anomaly_detection"
TYPE_CLASSIFICATION = "classification"

TYPE_NAMES = {
    TYPE_PREDICTION: "prediction",
    TYPE_ANOMALY_DETECTION: "anomaly_detection",
    TYPE_CLASSIFICATION: "classification",
}

DATASET_TYPE_MAPPING = {
    "M4 Competition Dataset": TYPE_PREDICTION,
    "M5 Competition Dataset": TYPE_PREDICTION,
    "Monash Time Series Forecasting Archive": TYPE_PREDICTION,
    "ETT-small": TYPE_PREDICTION,
    "Electricity/ECL": TYPE_PREDICTION,
    "Traffic": TYPE_PREDICTION,
    "Weather": TYPE_PREDICTION,
    "Exchange-Rate": TYPE_PREDICTION,
    "ILI": TYPE_PREDICTION,
    "Yahoo Webscope S5": TYPE_ANOMALY_DETECTION,
    "NAB": TYPE_ANOMALY_DETECTION,
    "SMAP": TYPE_ANOMALY_DETECTION,
    "MSL": TYPE_ANOMALY_DETECTION,
    "SMD": TYPE_ANOMALY_DETECTION,
    "UCR Time Series Classification Archive": TYPE_CLASSIFICATION,
    "UEA/UCR Multivariate Time Series Classification Archive": TYPE_CLASSIFICATION,
    "PhysioNet": TYPE_CLASSIFICATION,
}

FOLDER_TO_DATASET_NAME = {
    "Monash_Time_Series_Forecasting_Archive": "Monash Time Series Forecasting Archive",
    "ETT-small": "ETT-small",
    "ElectricityECL": "Electricity/ECL",
    "Exchange_Rate": "Exchange-Rate",
    "NAB": "NAB",
    "Traffic": "Traffic",
    "Weather": "Weather",
    "UEA&UCR_Multivariate_Time_Series_Classification_Archive": "UEA/UCR Multivariate Time Series Classification Archive",
}

def get_dataset_type(folder_name: str) -> str | None:
    if folder_name.startswith("__") or folder_name.startswith("."):
        return None
    
    dataset_name = FOLDER_TO_DATASET_NAME.get(folder_name, folder_name)
    dataset_type = DATASET_TYPE_MAPPING.get(dataset_name)
    
    if dataset_type is None:
        folder_lower = folder_name.lower()
        if "classification" in folder_lower or "class" in folder_lower:
            return TYPE_CLASSIFICATION
        elif "anomaly" in folder_lower:
            return TYPE_ANOMALY_DETECTION
        else:
            return TYPE_PREDICTION
    
    return dataset_type
