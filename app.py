import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Bank Marketing Analysis", layout="wide")

st.title("Bank Marketing Analysis with Streamlit")

# Veri seti CSV formatında okunur, ayraç olarak noktalı virgül kullanılır
@st.cache_data
def load_data():
    return pd.read_csv("bank.csv", sep=";")

df = load_data()

st.sidebar.header("Kontroller")
test_size = st.sidebar.slider("Test oranı", 0.2, 0.4, 0.3)
apply_pca = st.sidebar.checkbox("PCA uygula")
n_components = st.sidebar.slider("PCA bileşen sayısı", 2, 10, 2)

st.header("1️⃣ Veri Seti Genel Görünümü")
st.write("İlk 5 gözlem:")
st.dataframe(df.head())
st.write("Veri seti boyutu:", df.shape)

st.header("2️⃣ Veri Görselleştirme")

col1, col2 = st.columns(2)

with col1:
    # Hedef değişkenin sınıf dağılımı countplot ile görselleştirilir
    fig, ax = plt.subplots()
    sns.countplot(x="y", data=df, ax=ax)
    st.pyplot(fig)

with col2:
    # Yaş değişkeninin dağılımı histogram ile incelenir
    fig, ax = plt.subplots()
    sns.histplot(df["age"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

st.subheader("Sayısal Değişkenler için Korelasyon Isı Haritası")

# Korelasyon analizi yalnızca sayısal değişkenler için yapılır
# Bu nedenle kategorik değişkenler veri setinden çıkarılır
numeric_df = df.select_dtypes(include=["int64", "float64"])

# Pearson korelasyon katsayısı kullanılarak ısı haritası oluşturulur
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.header("3️⃣ Veri Ön İşleme")

# Kategorik değişkenler Label Encoding yöntemi ile sayısal değerlere dönüştürülür
encoded_df = df.copy()
le = LabelEncoder()
for col in encoded_df.select_dtypes(include="object"):
    encoded_df[col] = le.fit_transform(encoded_df[col])

# Bağımsız değişkenler (X) ve hedef değişken (y) ayrılır
X = encoded_df.drop("y", axis=1)
y = encoded_df["y"]

# Değişkenlerin aynı ölçekte olması için StandardScaler ile ölçekleme yapılır
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.success("Veri ön işleme adımları tamamlandı.")

if apply_pca:
    st.header("4️⃣ Principal Component Analysis (PCA)")

    # PCA, ölçeklenmiş veriye uygulanır
    # Amaç, varyansın büyük kısmını koruyarak boyutu azaltmaktır
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Her bileşenin açıkladığı varyans oranı görselleştirilir
    st.write("Açıklanan varyans oranları:")
    st.bar_chart(pca.explained_variance_ratio_)
else:
    # PCA uygulanmazsa ölçeklenmiş veri doğrudan modele verilir
    X_pca = X_scaled

st.header("5️⃣ Makine Öğrenmesi Modeli")

# Veri seti, eğitim ve test verisi olacak şekilde ikiye ayrılır
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=test_size, random_state=42
)

# Lojistik Regresyon modeli oluşturulur ve eğitim verisi ile eğitilir
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapılır
y_pred = model.predict(X_test)

# Modelin doğruluğu accuracy metriği ile hesaplanır
acc = accuracy_score(y_test, y_pred)
st.metric("Model Doğruluğu (Accuracy)", round(acc, 3))

st.subheader("Confusion Matrix")

# Confusion matrix ile modelin doğru ve yanlış tahminleri gösterilir
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

st.success("Model eğitimi ve değerlendirme süreci tamamlandı.")