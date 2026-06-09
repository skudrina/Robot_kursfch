import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import linprog

st.set_page_config(page_title="ИАС Производитель роботов", layout="wide")
st.title("🏭 Информационно-аналитическая система")
st.subheader("Производитель промышленных роботов")

CATEGORIES = [
    "Грузоподъемность_кг",
    "Точность_мм",
    "Досягаемость_мм",
    "Степени_свободы",
    "Безопасность"
]

def load_data():
    try:
        df = pd.read_csv("robots.csv", encoding="utf-8")
        # Приводим названия колонок к единому виду (на случай, если CSV другой)
        df.columns = [col.strip().replace(" ", "_") for col in df.columns]
        return df
    except:
        data = {
            "Производитель": ["Аркодим", "Промобот", "Завод Роботов"],
            "Грузоподъемность_кг": [8, 5, 9],
            "Точность_мм": [7, 9, 6],
            "Досягаемость_мм": [6, 5, 9],
            "Степени_свободы": [9, 6, 8],
            "Безопасность": [5, 9, 4]
        }
        return pd.DataFrame(data)

if 'df' not in st.session_state:
    st.session_state.df = load_data()
if 'manual_df' not in st.session_state:
    st.session_state.manual_df = pd.DataFrame(columns=["Производитель"] + CATEGORIES)

st.sidebar.header("📁 Управление данными")
data_source = st.sidebar.radio(
    "Источник данных",
    ["Предустановленные данные", "Загрузить CSV/JSON", "Ручной ввод"]
)

if data_source == "Загрузить CSV/JSON":
    uploaded_file = st.sidebar.file_uploader("Выберите файл", type=["csv", "json"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            new_df = pd.read_csv(uploaded_file, encoding="utf-8")
        else:
            new_df = pd.read_json(uploaded_file)
        # Очистим названия колонок
        new_df.columns = [col.strip().replace(" ", "_") for col in new_df.columns]
        if all(cat in new_df.columns for cat in CATEGORIES) and "Производитель" in new_df.columns:
            st.session_state.df = new_df
            st.sidebar.success("Данные загружены!")
        else:
            st.sidebar.error("Файл должен содержать колонки: Производитель и " + ", ".join(CATEGORIES))

elif data_source == "Ручной ввод":
    st.sidebar.subheader("➕ Добавить производителя")
    new_name = st.sidebar.text_input("Название производителя")
    new_vals = []
    for cat in CATEGORIES:
        val = st.sidebar.slider(f"{cat}", 1, 10, 5, key=f"manual_{cat}")
        new_vals.append(val)
    if st.sidebar.button("Добавить в список"):
        if new_name:
            new_row = {"Производитель": new_name}
            for i, cat in enumerate(CATEGORIES):
                new_row[cat] = new_vals[i]
            st.session_state.manual_df = pd.concat(
                [st.session_state.manual_df, pd.DataFrame([new_row])],
                ignore_index=True
            )
            st.sidebar.success(f"Добавлен {new_name}")
            st.rerun()
    
    display_df = pd.concat([st.session_state.df, st.session_state.manual_df], ignore_index=True)
else:
    display_df = st.session_state.df

if data_source != "Ручной ввод":
    display_df = st.session_state.df

def calculate_technical_level(row, weights):
    score = 0
    for cat in CATEGORIES:
        score += row[cat] * weights[cat]
    return round(score, 2)

st.header("⚖️ Настройка весовых коэффициентов")
cols = st.columns(len(CATEGORIES))
weights = {}
for i, cat in enumerate(CATEGORIES):
    weights[cat] = cols[i].number_input(f"{cat}", 0.0, 1.0, 0.2, step=0.05)

weight_sum = sum(weights.values())
st.info(f"Сумма весов: {weight_sum:.2f} (должна быть равна 1.0)")

if abs(weight_sum - 1.0) > 0.01:
    st.error("❌ Сумма весов не равна 1.0! Расчет заблокирован.")
else:
    display_df_copy = display_df.copy()
    display_df_copy["Технический уровень"] = display_df_copy.apply(
        lambda row: calculate_technical_level(row, weights), axis=1
    )
    display_df_copy = display_df_copy.sort_values("Технический уровень", ascending=False)
    
    tab1, tab2, tab3 = st.tabs(["📊 Рейтинг и сравнение", "📈 Прогноз динамики", "⚙️ Оптимизация производства"])
    
    with tab1:
        st.subheader("🏆 Рейтинг производителей")
        st.dataframe(display_df_copy[["Производитель"] + CATEGORIES + ["Технический уровень"]])
        
        fig_bar = px.bar(
            display_df_copy, x="Производитель", y="Технический уровень",
            title="Интегральный технический уровень",
            color="Технический уровень", color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.subheader("📡 Радиальные профили")
        top_5 = display_df_copy.head(5)
        fig_radar = go.Figure()
        for _, row in top_5.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[cat] for cat in CATEGORIES],
                theta=CATEGORIES,
                fill='toself',
                name=row["Производитель"]
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])))
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.subheader("🔥 Тепловая карта характеристик")
        heat_data = display_df_copy[CATEGORIES].T
        heat_labels = display_df_copy["Производитель"].tolist()
        fig_heat = px.imshow(
            heat_data, text_auto=True, aspect="auto",
            x=heat_labels, y=CATEGORIES,
            color_continuous_scale="RdYlGn", range_color=[1,10],
            title="Матрица оценок (красный — слабо, зелёный — сильно)"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Прогноз динамики показателей (на примере грузоподъёмности)")
        st.markdown("Модель полиномиальной регрессии с автоматическим выбором степени (1–3) для минимизации RMSE на тесте.")
        
        years = np.array([2018, 2019, 2020, 2021, 2022, 2023, 2024])
        avg_payload = display_df["Грузоподъемность_кг"].mean()
        values = avg_payload + 0.5 * (years - 2018) + np.random.normal(0, 0.3, len(years))
        
        horizon = st.slider("Горизонт прогноза (лет)", 1, 5, 3)
        
        X = years.reshape(-1, 1)
        y = values
        
        best_degree = 1
        best_test_rmse = float('inf')
        train_rmse_list = []
        test_rmse_list = []
        
        for d in range(1, 4):
            poly = PolynomialFeatures(degree=d, include_bias=False)
            X_poly = poly.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            train_rmse_list.append(train_rmse)
            test_rmse_list.append(test_rmse)
            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                best_degree = d
        # Построение кривой валидации
        degrees = [1, 2, 3]
        fig_validation = go.Figure()
        fig_validation.add_trace(go.Scatter(x=degrees, y=train_rmse_list, mode='lines+markers', name='RMSE обучения', line=dict(color='blue')))
        fig_validation.add_trace(go.Scatter(x=degrees, y=test_rmse_list, mode='lines+markers', name='RMSE теста', line=dict(color='red')))
        fig_validation.update_layout(
            title="Кривая валидации модели (зависимость RMSE от степени полинома)",
            xaxis_title="Степень полинома",
            yaxis_title="RMSE",
            legend_title="Тип ошибки"
        )
        st.plotly_chart(fig_validation, use_container_width=True)
        
        st.write(f"**Выбранная степень полинома: {best_degree}** (минимизирует RMSE на тесте)")
        
        poly_final = PolynomialFeatures(degree=best_degree, include_bias=False)
        X_poly_final = poly_final.fit_transform(X)
        model_final = LinearRegression()
        model_final.fit(X_poly_final, y)
        
        future_years = np.arange(years[-1]+1, years[-1]+horizon+1)
        X_future = future_years.reshape(-1, 1)
        X_future_poly = poly_final.transform(X_future)
        predictions = model_final.predict(X_future_poly)
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=years, y=y, mode='lines+markers', name='Исторические данные'))
        fig_forecast.add_trace(go.Scatter(x=future_years, y=predictions, mode='lines+markers', name='Прогноз'))
        fig_forecast.update_layout(title=f"Прогноз средней грузоподъёмности (степень {best_degree})",
                                   xaxis_title="Год", yaxis_title="Грузоподъемность, кг")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.metric("RMSE на обучении", f"{train_rmse_list[best_degree-1]:.3f}")
        st.metric("RMSE на тесте", f"{test_rmse_list[best_degree-1]:.3f}")
    
    with tab3:
        st.subheader("⚙️ Оптимизация производственной программы")
        st.markdown("Максимизация прибыли при выпуске **тяжёлых роботов (ТР)** и **коллаборативных роботов (КР)**.")
        
        col1, col2 = st.columns(2)
        with col1:
            price_heavy = st.number_input("Цена тяжёлого робота (у.е.)", value=300, step=10)
            time_heavy = st.number_input("Время на тяжёлого робота (ч)", value=2.0, step=0.5)
            mat_heavy = st.number_input("Материалы на тяжёлого робота (м²)", value=1.5, step=0.5)
        with col2:
            price_collab = st.number_input("Цена коллаборативного робота (у.е.)", value=220, step=10)
            time_collab = st.number_input("Время на коллаборативного робота (ч)", value=4.0, step=0.5)
            mat_collab = st.number_input("Материалы на коллаборативного робота (м²)", value=1.0, step=0.5)
        
        limit_time = st.number_input("Лимит времени (ч)", value=1518.0)
        limit_mat = st.number_input("Лимит материалов (м²)", value=2000.0)
        
        if st.button("Рассчитать оптимальный план"):
            c = [-price_heavy, -price_collab]
            A = [[time_heavy, time_collab],
                 [mat_heavy, mat_collab]]
            b = [limit_time, limit_mat]
            bounds = [(0, None), (0, None)]
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.success:
                x1, x2 = res.x
                profit = -res.fun
                st.success(f"✅ Оптимальный план:\n- Тяжёлых роботов: {x1:.0f} ед.\n- Коллаборативных роботов: {x2:.0f} ед.\n- Максимальная прибыль: {profit:.2f} у.е.")
            else:
                st.error("Не удалось найти решение. Проверьте ограничения.")