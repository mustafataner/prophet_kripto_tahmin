import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.write("""
# kripto fiyat tahmin uygulaması
**KULLANIM: BTC-ETH-DOT-LINK usd paritlerinden birini seçerek prophet tahminleme modeli ile 2 ay sonraki kripto para birimini tahmin etmek için yazılmıştır.

**YASAL UYARI:** Bu uygulama yalnızca deneme amaçlıdır. Yatırım tavsiyesi değildir. Bu uygulamada sunulan verilerin doğruluğu veya eksiksizliği konusunda herhangi bir sorumluluk kabul edilmemektedir. Lütfen yatırım kararları vermeden önce uzman bir danışmana başvurunuz.
""")


st.title('kripto fiyat tahmin uygulaması')

stocks = ('BTC-USD', 'ETH-USD',  'DOT-USD','LINK-USD')
selected_stock = st.selectbox('TAHMİN İÇİN VERİ SETİNİ SEÇİNİZ', stocks)

# selected_stock = st.text_input("Hisse senedi sembolünü girin (Örn: AAPL):") böyle kullanıcı girecektir


n_years = 2
period = 60


@st.cache_data(ttl=60*5)
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('veriler yükleniyor...')
data = load_data(selected_stock)
data_load_state.text('veriler yükleniyor... başarılı!')

st.subheader('geçmiş veriler ')
st.write(data.tail())


# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open", line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close", line=dict(color='green')))
    fig.layout.update(title_text='ZAMAN SERİSİ VERİLERİ', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    st.plotly_chart(fig, use_container_width=True)


plot_raw_data()


df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)


st.subheader('TAHMİN VERİLERİ')
st.write(forecast.tail())

st.write(f" {n_years} AYLIK TAHMİN GRAFİĞİ \n(kırmızı çizgiler tahmin, mavi çizgiler gerçek değerleridir.)")
fig1 = plot_plotly(m, forecast)
fig1.update_traces(line=dict(color='red'), marker=dict(color='blue'))
st.plotly_chart(fig1)


st.write("tahmin bileşenleri")
fig2 = m.plot_components(forecast)
st.write(fig2)
