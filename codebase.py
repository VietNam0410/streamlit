import pandas as pd
import mlflow

# Đặt URI để lưu trữ kết quả MLFlow trong thư mục "mlruns"

# Định nghĩa một experiment để quản lý quá trình tracking
mlflow.set_experiment("Titanic_Experiment")

# Đọc tập dữ liệu Titanic
file_path = "titanic.csv"
df = pd.read_csv(file_path)

# Hiển thị thông tin dữ liệu
print(df.head())
from sklearn.preprocessing import LabelEncoder

# Bắt đầu một run trong MLFlow
with mlflow.start_run(run_name="Data_Preprocessing"):
    mlflow.log_param("Initial_Rows", df.shape[0])
    mlflow.log_param("Initial_Columns", df.shape[1])

    # Xóa cột không cần thiết
    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

    # Xóa các dòng có giá trị thiếu
    df.dropna(inplace=True)

    # Mã hóa dữ liệu phân loại
    encoder = LabelEncoder()
    df['Sex'] = encoder.fit_transform(df['Sex'])
    df['Embarked'] = encoder.fit_transform(df['Embarked'].astype(str))

    # Lưu kích thước dữ liệu sau xử lý
    mlflow.log_param("Processed_Rows", df.shape[0])
    mlflow.log_param("Processed_Columns", df.shape[1])

    # Log dữ liệu mẫu
    mlflow.log_text(df.head().to_string(), "sample_data.txt")

    print("Dữ liệu sau khi xử lý:")
    print(df.head())
    from sklearn.model_selection import train_test_split

# Tách đặc trưng (X) và nhãn (y)
X = df.drop(columns=['Survived'])
y = df['Survived']

# Chia tập Train / Validation / Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Logging thông tin tập dữ liệu
with mlflow.start_run(run_name="Data_Splitting"):
    mlflow.log_param("Train_Size", len(X_train))
    mlflow.log_param("Validation_Size", len(X_valid))
    mlflow.log_param("Test_Size", len(X_test))

print(f"Train size: {len(X_train)}, Valid size: {len(X_valid)}, Test size: {len(X_test)}")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Bắt đầu logging quá trình huấn luyện mô hình
with mlflow.start_run(run_name="Model_Training"):
    # Khởi tạo mô hình Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Thực hiện Cross Validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    # Log hyperparameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("cv_folds", 5)

    # Log kết quả đánh giá
    mlflow.log_metric("cv_accuracy_mean", np.mean(scores))
    mlflow.log_metric("cv_accuracy_std", np.std(scores))

    print(f"Cross Validation Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # Huấn luyện mô hình trên toàn bộ tập Train + Validation
    model.fit(X_train, y_train)

    # Lưu mô hình vào MLFlow
    mlflow.sklearn.log_model(model, "random_forest_model")

from sklearn.metrics import accuracy_score

with mlflow.start_run(run_name="Model_Evaluation"):
    # Dự đoán trên tập Test
    y_pred = model.predict(X_test)

    # Tính độ chính xác
    test_accuracy = accuracy_score(y_test, y_pred)

    # Log kết quả
    mlflow.log_metric("test_accuracy", test_accuracy)

    print(f"Test Accuracy: {test_accuracy:.4f}")

