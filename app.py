import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Đọc dữ liệu Titanic
@st.cache_data
def load_data():
    file_path = "titanic.csv"
    df = pd.read_csv(file_path)
    return df

# Tiền xử lý dữ liệu
def preprocess_data(df):
    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    df.dropna(inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].astype('category').cat.codes
    return df

# Huấn luyện mô hình Random Forest
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    model.fit(X_train, y_train)
    return model, scores

# Giao diện Streamlit
st.title("🚢 Titanic Survival Prediction")
st.write("Minh họa kết quả xử lý dữ liệu, huấn luyện và đánh giá mô hình.")

# Load dữ liệu
df = load_data()
st.subheader("📊 Dữ liệu ban đầu")
st.write(df.head())

# Xử lý dữ liệu
df = preprocess_data(df)

# Chia tập dữ liệu
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hiển thị thông tin tập dữ liệu
st.subheader("📌 Thống kê dữ liệu sau xử lý")
st.write(f"👨‍🏫 Số dòng: {df.shape[0]}, Số cột: {df.shape[1]}")
st.write(df.head())

# Huấn luyện mô hình
st.subheader("🧠 Huấn luyện mô hình Random Forest")
with st.spinner("Đang huấn luyện..."):
    model, scores = train_model(X_train, y_train)
    joblib.dump(model, "random_forest_model.pkl")

# Hiển thị kết quả Cross Validation
st.write(f"📈 Độ chính xác trung bình (CV): {scores.mean():.4f} ± {scores.std():.4f}")

# Dự đoán trên tập Test
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
st.write(f"✅ Độ chính xác trên tập Test: {test_accuracy:.4f}")

# Vẽ biểu đồ quan trọng của các đặc trưng
st.subheader("🔍 Độ quan trọng của các đặc trưng")
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 4))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
st.pyplot(plt)

# Kết nối với MLFlow
st.subheader("📡 MLFlow Logging")
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Titanic_Experiment")

with mlflow.start_run(run_name="Streamlit_Model"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("cv_accuracy", scores.mean())
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")

st.success("🎯 Dữ liệu và mô hình đã được logging vào MLFlow!")

