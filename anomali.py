# Gerekli kütüphaneleri içe aktarıyoruz
import pandas as pd          # Veri işleme için
import numpy as np           # Sayısal hesaplamalar için
import streamlit as st       # Web arayüzü için
import plotly.express as px  # Grafik çizmek için
from sklearn.ensemble import IsolationForest  # Anomali tespit modeli

# Sayfanın başlığını ve alt yazısını ayarlıyoruz
st.title("🔍 Finansal Anomali Tespit Sistemi")
st.write("Intellium - AI Destekli Finansal Analiz")

# @st.cache_data → bu fonksiyon bir kez çalışır, sayfa yenilenince tekrar çalışmaz (hız için)
@st.cache_data
def veri_uret():
    np.random.seed(42)  # Aynı veriyi her seferinde üretmek için sabit tohum
    n = 300             # 300 adet normal işlem oluşturacağız

    # Normal işlemler: ortalama 1000 TL, 200 TL sapmayla rastgele tutar
    # Saat: 08:00-18:00 arası (mesai saatleri), gün: 1-7 arası
    df = pd.DataFrame({
        "islem_id": range(1, n+1),
        "tutar": np.random.normal(1000, 200, n),
        "saat": np.random.randint(8, 18, n),
        "gun": np.random.randint(1, 8, n)
    })

    # Anormal işlemler: 5000-10000 TL arası yüksek tutarlar
    # Gece 00:00-05:00 arası yapılmış → şüpheli!
    anomaliler = pd.DataFrame({
        "islem_id": range(n+1, n+16),
        "tutar": np.random.uniform(5000, 10000, 15),
        "saat": np.random.randint(0, 5, 15),
        "gun": np.random.randint(1, 8, 15)
    })

    # Normal ve anormal işlemleri birleştir, index'i sıfırla
    return pd.concat([df, anomaliler], ignore_index=True)

# Fonksiyonu çağır, veriyi al
df = veri_uret()

# Isolation Forest modeli: verinin %5'ini anomali olarak işaretle
# Bu model, "normal"den uzak olan noktaları tespit eder
model = IsolationForest(contamination=0.05, random_state=42)

# Modeli eğit ve tahmin yap → 1: Normal, -1: Anomali
df["anomali"] = model.fit_predict(df[["tutar", "saat", "gun"]])

# Sayısal değerleri okunabilir etikete çevir
df["durum"] = df["anomali"].map({1: "Normal", -1: "⚠️ Şüpheli"})

# Sayfayı 3 sütuna böl, her birine bir metrik koy
col1, col2, col3 = st.columns(3)
col1.metric("Toplam İşlem", len(df))
col2.metric("Normal İşlem", len(df[df["durum"] == "Normal"]))
col3.metric("Şüpheli İşlem", len(df[df["durum"] == "⚠️ Şüpheli"]))

# Scatter plot: X ekseni işlem ID, Y ekseni tutar
# Yeşil → Normal, Kırmızı → Şüpheli
fig = px.scatter(
    df,
    x="islem_id",
    y="tutar",
    color="durum",
    color_discrete_map={"Normal": "green", "⚠️ Şüpheli": "red"},
    title="İşlem Analizi — Kırmızılar Şüpheli",
    labels={"tutar": "İşlem Tutarı (TL)", "islem_id": "İşlem ID"}
)
st.plotly_chart(fig)  # Grafiği sayfaya çiz

# Sadece şüpheli işlemleri tablo olarak göster
st.subheader("⚠️ Şüpheli İşlemler")
st.dataframe(df[df["durum"] == "⚠️ Şüpheli"][["islem_id", "tutar", "saat", "gun"]])

