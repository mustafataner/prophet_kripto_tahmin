import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Genel ayarlar
START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Uygulama başlığı ve açıklaması
st.title("Borsa ve Kripto Fiyat Tahmin Uygulaması")
st.write("""
**KULLANIM:** Bu uygulama, kullanıcı tarafından girilen herhangi bir ticker sembolü üzerinden veri çekerek,
hem Prophet, ML/RL tabanlı (örnek ML modeli) tahmin hem de Monte Carlo simülasyonu ile gelecekteki fiyat tahminleri sunar.

**YASAL UYARI:** Bu uygulama yalnızca deneme amaçlıdır. Yatırım tavsiyesi değildir.
Lütfen yatırım kararları vermeden önce uzman bir danışmana başvurunuz.
""")

# Sayfa seçimi (multi-page yapı)
page = st.sidebar.radio("Sayfa Seçiniz", 
                          ["Prophet Tahmin", "ML/RL Tahmin (Örnek ML Model)", "Monte Carlo Simülasyonu"])

@st.cache_data(ttl=60*5)
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

if page == "Prophet Tahmin":
    st.header("Prophet Tahmin Sayfası")
    ticker = st.text_input("Lütfen tahmin için ticker sembolünü girin (örn: BTC-USD, AAPL, ISBANK.IS):", key="prophet")
    
    if ticker:
        # Veri çekme
        data_load_state = st.text('Veriler yükleniyor...')
        data = load_data(ticker)
        data_load_state.text('Veriler yüklendi!')

        st.subheader("Geçmiş Veriler")
        st.write(data.tail())

        # Zaman serisi grafik
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Açılış Fiyatı", line=dict(color='red')))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Kapanış Fiyatı", line=dict(color='green')))
            fig.update_layout(title='Zaman Serisi Verileri', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)

        plot_raw_data()

        # Prophet için veri hazırlığı
        df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

        # Prophet Modeli Eğitimi ve Tahmin
        m = Prophet()
        m.fit(df_train)
        # 60 gün (2 ay) sonrası tahmin ediliyor
        period = 60  
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        st.subheader("Tahmin Verileri (Son Kısımlar)")
        st.write(forecast.tail())

        st.write("2 Aylık Tahmin Grafiği \n(Kırmızı çizgi: Tahmin, Mavi çizgi: Gerçek Değerler)")
        fig1 = plot_plotly(m, forecast)
        fig1.update_traces(line=dict(color='red'), marker=dict(color='blue'))
        st.plotly_chart(fig1, use_container_width=True)

        st.write("Tahmin Bileşenleri")
        fig2 = m.plot_components(forecast)
        st.write(fig2)

elif page == "ML/RL Tahmin (Örnek ML Model)":
    st.header("ML/RL Tahmin Sayfası (Örnek ML Model)")
    ticker_ml = st.text_input("Lütfen tahmin için ticker sembolünü girin (örn: BTC-USD, AAPL, ISBANK.IS):", key="ml")
    forecast_period = st.number_input("Tahmin dönemi (gün):", min_value=1, value=60)

    if ticker_ml:
        # Veri çekme
        data_load_state = st.text('Veriler yükleniyor...')
        data_ml = load_data(ticker_ml)
        data_load_state.text('Veriler yüklendi!')

        st.subheader("Geçmiş Veriler")
        st.write(data_ml.tail())

        # ML Modeli için veri hazırlığı: Kapanış fiyatını kullanarak lag özellikleri oluşturma
        df_ml = data_ml[['Date', 'Close']].copy()
        df_ml['Date'] = pd.to_datetime(df_ml['Date'])
        df_ml.set_index('Date', inplace=True)
        
        # Lag özellikleri: Son 5 gün
        for lag in range(1, 6):
            df_ml[f'lag_{lag}'] = df_ml['Close'].shift(lag)
        df_ml = df_ml.dropna()

        st.write("ML Modeli için oluşturulan veri:")
        st.write(df_ml.tail())

        # Model eğitimi: Basit bir RandomForestRegressor
        feature_cols = [f'lag_{lag}' for lag in range(1, 6)]
        X = df_ml[feature_cols]
        y = df_ml['Close']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Iteratif tahmin: Son 5 günü kullanarak ileriye doğru tahmin
        last_obs = df_ml['Close'][-5:].values.tolist()
        predictions = []
        for i in range(forecast_period):
            input_features = np.array(last_obs[-5:]).reshape(1, -1)
            pred = model.predict(input_features)[0]
            predictions.append(pred)
            last_obs.append(pred)

        # Tahmin sonuçlarını DataFrame'e aktarma
        last_date = df_ml.index[-1]
        forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_period)]
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Close': predictions})

        # Gerçek ve tahmin grafiklerinin çizilmesi
        fig_ml = go.Figure()
        fig_ml.add_trace(go.Scatter(x=df_ml.index, y=df_ml['Close'], mode='lines', name='Gerçek Fiyat'))
        fig_ml.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted Close'], mode='lines', name='Tahmin'))
        fig_ml.update_layout(title='ML Tahmin Grafiği', xaxis_title='Tarih', yaxis_title='Fiyat')
        st.plotly_chart(fig_ml, use_container_width=True)

        st.subheader("Tahmin Verileri")
        st.write(forecast_df)
        st.write("Not: Bu ML modeli, gerçek piyasa uygulamalarında kullanılan ileri düzey ML/RL modellerinin basitleştirilmiş örneğidir.")

elif page == "Monte Carlo Simülasyonu":
    st.header("Monte Carlo Simülasyonu Sayfası")
    ticker_mc = st.text_input("Lütfen simülasyon için ticker sembolünü girin (örn: BTC-USD, AAPL, ISBANK.IS):", key="mc")
    num_simulations = st.number_input("Simülasyon sayısı:", min_value=10, value=100)
    forecast_days = st.number_input("Tahmin dönemi (gün):", min_value=1, value=60)

    if ticker_mc:
        # Veri çekme
        data_load_state = st.text('Veriler yükleniyor...')
        data_mc = load_data(ticker_mc)
        data_load_state.text('Veriler yüklendi!')

        st.subheader("Geçmiş Veriler")
        st.write(data_mc.tail())

        # Monte Carlo Simülasyonu için veri hazırlığı
        # Kapanış fiyatları üzerinden günlük getiriler hesaplanıyor
        data_mc['Return'] = data_mc['Close'].pct_change()
        data_mc = data_mc.dropna()
        last_price = data_mc['Close'].iloc[-1]
        
        # Drift ve volatilite hesaplamaları
        drift = data_mc['Return'].mean()
        volatility = data_mc['Return'].std()

        st.write(f"Drift: {drift:.5f} - Volatilite: {volatility:.5f}")

        # Simülasyonları gerçekleştir
        simulation_df = pd.DataFrame()
        for i in range(int(num_simulations)):
            prices = [last_price]
            for j in range(forecast_days):
                shock = np.random.normal(loc=drift, scale=volatility)
                price = prices[-1] * (1 + shock)
                prices.append(price)
            simulation_df[f"Sim_{i+1}"] = prices

        # Tarih indeksini oluşturma
        forecast_index = [data_mc['Date'].iloc[-1] + pd.Timedelta(days=i) for i in range(forecast_days+1)]
        simulation_df.index = forecast_index

        # Grafik çizimi
        fig_mc = go.Figure()
        for col in simulation_df.columns:
            fig_mc.add_trace(go.Scatter(x=simulation_df.index, y=simulation_df[col], mode='lines', name=col))
        fig_mc.update_layout(title='Monte Carlo Simülasyonu: Fiyat Yolları', xaxis_title='Tarih', yaxis_title='Fiyat')
        st.plotly_chart(fig_mc, use_container_width=True)

        st.subheader("Simülasyon Sonuçları")
        st.write(simulation_df.tail())
