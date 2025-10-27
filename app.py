# coding=utf-8
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import numpy as np
import warnings
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

warnings.filterwarnings("ignore")


def setup_mlflow():
    """设置MLflow跟踪"""
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if mlflow_tracking_uri and mlflow_username and mlflow_password:
        try:
            from dagshub import DAGsHub

            # 连接到DAGsHub
            repo_owner = (
                mlflow_username.split("/")[-1]
                if "/" in mlflow_username
                else mlflow_username
            )
            DAGsHub(
                repo_owner=repo_owner,
                repo_name="titanic-survival-prediction",
                mlflow=True,
            )
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            print(f"成功连接到 MLflow 跟踪服务器: {mlflow_tracking_uri}")
        except ImportError:
            print("DAGsHub 未安装，使用本地 MLflow 跟踪")
            mlflow.set_tracking_uri("./mlruns")
    else:
        print("使用本地 MLflow 跟踪")
        mlflow.set_tracking_uri("./mlruns")

    mlflow.set_experiment("Titanic Survival Prediction")
    return mlflow


def main():
    # 设置MLflow
    mlflow = setup_mlflow()

    # 开始MLflow运行
    with mlflow.start_run():
        print("开始泰坦尼克号生存预测...")

        # 记录参数
        mlflow.log_param("test_size", 0.3)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("model_type", "SVM")
        mlflow.log_param("kernel", "rbf")
        mlflow.log_param("C", 1)
        mlflow.log_param("gamma", 0.1)

        # 读取数据集
        df = pd.read_csv("data/train_and_test2.csv", encoding="utf-8")
        print(f"数据集形状: {df.shape}")

        # 记录数据集信息
        mlflow.log_param("dataset_rows", df.shape[0])
        mlflow.log_param("dataset_columns", df.shape[1])

        # 删除zero列
        zero_columns = [col for col in df.columns if col.startswith("zero")]
        df.drop(zero_columns, inplace=True, axis=1)
        print(f"删除zero列后形状: {df.shape}")

        # 对缺失值进行向前填充
        if "Embarked" in df.columns:
            df["Embarked"].fillna(method="ffill", inplace=True)

        # 建立标签
        X = df.drop(["Passengerid", "2urvived"], axis=1)
        y = df["2urvived"]
        print(f"特征数量: {X.shape[1]}")

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

        # 数据标准化
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)

        # 实例化SVM
        svm = SVC(C=1, gamma=0.1, kernel="rbf", random_state=42)
        svm.fit(X_train, y_train)

        y_pred = svm.predict(X_test)

        # 计算混淆矩阵
        confusion = confusion_matrix(y_test, y_pred)
        print("原始混淆矩阵：")
        print(confusion)

        # 归一化混淆矩阵
        confusion_normalized = (
            confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
        )
        print("\n归一化混淆矩阵：")
        print(confusion_normalized)

        # 计算模型评价指标
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(
            f"\naccuracy={accuracy:.4f}, recall={recall:.4f}, precision={precision:.4f}, f1={f1:.4f}"
        )

        # 记录指标到MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)

        # 记录模型
        mlflow.sklearn.log_model(svm, "svm_model")

        # 记录混淆矩阵图像
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        # 保存图像到临时文件并记录到MLflow
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # 打印详细的分类报告
        print("\n详细分类报告：")
        report = classification_report(y_test, y_pred, target_names=["负类", "正类"])
        print(report)

        # 记录分类报告
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        print("MLflow运行完成！运行ID:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
