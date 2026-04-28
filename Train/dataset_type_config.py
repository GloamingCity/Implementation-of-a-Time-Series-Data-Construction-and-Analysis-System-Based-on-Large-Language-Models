
"""
数据集类型配置文件
根据 Data_Collection.xlsx 中的任务描述自动归类
"""
from pathlib import Path

# 三大类型定义
TYPE_PREDICTION = "prediction"
TYPE_ANOMALY_DETECTION = "anomaly_detection"
TYPE_CLASSIFICATION = "classification"

# 类型名称映射（用于文件夹命名）
TYPE_NAMES = {
    TYPE_PREDICTION: "prediction",
    TYPE_ANOMALY_DETECTION: "anomaly_detection",
    TYPE_CLASSIFICATION: "classification",
}

# 现有数据集的类型映射（根据 Data_Collection.xlsx 推断）
DATASET_TYPE_MAPPING = {
    # 预测类
    "M4 Competition Dataset": TYPE_PREDICTION,
    "M5 Competition Dataset": TYPE_PREDICTION,
    "Monash Time Series Forecasting Archive": TYPE_PREDICTION,
    "ETT-small": TYPE_PREDICTION,
    "Electricity/ECL": TYPE_PREDICTION,
    "Traffic": TYPE_PREDICTION,
    "Weather": TYPE_PREDICTION,
    "Exchange-Rate": TYPE_PREDICTION,
    "ILI": TYPE_PREDICTION,
    
    # 异常检测类
    "Yahoo Webscope S5": TYPE_ANOMALY_DETECTION,
    "NAB": TYPE_ANOMALY_DETECTION,
    "SMAP": TYPE_ANOMALY_DETECTION,
    "MSL": TYPE_ANOMALY_DETECTION,
    "SMD": TYPE_ANOMALY_DETECTION,
    
    # 分类类
    "UCR Time Series Classification Archive": TYPE_CLASSIFICATION,
    "UEA/UCR Multivariate Time Series Classification Archive": TYPE_CLASSIFICATION,
    "PhysioNet": TYPE_CLASSIFICATION,
}

# 文件夹名称到数据集名称的映射（根据实际文件夹名调整）
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
    """
    根据文件夹名称获取数据集类型
    返回 None 表示跳过该文件夹（如缓存文件夹）
    """
    # 跳过缓存文件夹
    if folder_name.startswith("__") or folder_name.startswith("."):
        return None
    
    # 先查找直接映射
    dataset_name = FOLDER_TO_DATASET_NAME.get(folder_name, folder_name)
    
    # 再查找类型映射
    dataset_type = DATASET_TYPE_MAPPING.get(dataset_name)
    
    # 如果没有找到，尝试根据关键词推断
    if dataset_type is None:
        folder_lower = folder_name.lower()
        if "classification" in folder_lower or "class" in folder_lower:
            return TYPE_CLASSIFICATION
        elif "anomaly" in folder_lower:
            return TYPE_ANOMALY_DETECTION
        else:
            # 默认归为预测类
            return TYPE_PREDICTION
    
    return dataset_type
