import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
from typing import Dict, List

# Конфигурация страницы
st.set_page_config(
    page_title="Агропайплайн - Анализ сельскохозяйственных культур",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    
    .recommendation-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FFC107;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #DC3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Конфигурация API
API_BASE_URL = "http://localhost:8000"

# @st.cache_data  # ОТКЛЮЧАЕМ КЕШИРОВАНИЕ для исправления проблемы консистентности
def call_api(analysis_type: str, image_file) -> Dict:
    """Вызов API для анализа изображения"""
    try:
        # Выбираем endpoint в зависимости от типа анализа
        endpoint_map = {
            "Комплексный анализ": "analyze",
            "Классификация культуры": "classify", 
            "Оценка качества": "quality",
            "Прогноз урожайности": "yield"
        }
        
        endpoint = endpoint_map.get(analysis_type)
        if not endpoint:
            return {
                "success": False,
                "error": "Неизвестный тип анализа",
                "detail": f"Тип анализа '{analysis_type}' не поддерживается"
            }
        
        # Подготавливаем файл
        image_file.seek(0)
        file_bytes = image_file.read()
        image_file.seek(0)  # Возвращаем указатель в начало
        
        # Проверяем, что файл не пустой
        if len(file_bytes) == 0:
            return {
                "success": False,
                "error": "Пустой файл",
                "detail": "Загруженный файл пустой или поврежден"
            }
        
        # Пытаемся открыть как изображение для предварительной проверки
        try:
            Image.open(io.BytesIO(file_bytes)).verify()
        except Exception as img_error:
            return {
                "success": False,
                "error": "Неверный формат изображения",
                "detail": f"Файл не является корректным изображением: {str(img_error)}"
            }
        
        # Подготавливаем файл для отправки
        files = {
            "file": (
                image_file.name, 
                file_bytes, 
                f"image/{image_file.name.split('.')[-1].lower()}"
            )
        }
        
        response = requests.post(
            f"{API_BASE_URL}/{endpoint}", 
            files=files,
            timeout=30  # Добавляем тайм-аут
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            # Пытаемся парсить JSON ошибку
            try:
                error_detail = response.json()
                return {
                    "success": False,
                    "error": f"Ошибка API: {response.status_code}",
                    "detail": error_detail.get("detail", response.text)
                }
            except:
                return {
                    "success": False,
                    "error": f"Ошибка API: {response.status_code}",
                    "detail": response.text
                }
                
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Тайм-аут запроса",
            "detail": "Сервер слишком долго не отвечает. Попробуйте позже."
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Не удалось подключиться к серверу",
            "detail": "Убедитесь, что API сервер запущен на localhost:8000"
        }
    except Exception as e:
        return {
            "success": False,
            "error": "Ошибка при вызове API",
            "detail": str(e)
        }

def display_image_info(image: Image.Image):
    """Отображение информации об изображении"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Ширина", f"{image.width} px")
    
    with col2:
        st.metric("Высота", f"{image.height} px")
    
    with col3:
        st.metric("Формат", image.format or "Неизвестен")

def display_classification_results(result: Dict):
    """Отображение результатов классификации"""
    if not result.get("success", False):
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ Ошибка классификации</h4>
            <p><strong>Ошибка:</strong> {result.get('error', 'Неизвестная ошибка')}</p>
            <p><strong>Детали:</strong> {result.get('detail', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Основная информация
    confidence = result.get("confidence", 0)
    predicted_class_ru = result.get("predicted_class_ru", "")
    is_confident = result.get("is_confident", False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_color = "success" if is_confident else "warning"
        st.markdown(f"""
        <div class="{confidence_color}-box">
            <h3>🌾 Определенная культура</h3>
            <h2>{predicted_class_ru}</h2>
            <p><strong>Уверенность:</strong> {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # График вероятностей
        probabilities_ru = result.get("probabilities_ru", {})
        if probabilities_ru:
            fig = px.bar(
                x=list(probabilities_ru.values()),
                y=list(probabilities_ru.keys()),
                orientation='h',
                title="Вероятности классов",
                labels={'x': 'Вероятность', 'y': 'Культура'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def display_quality_results(result: Dict):
    """Отображение результатов оценки качества"""
    if not result.get("success", False):
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ Ошибка оценки качества</h4>
            <p><strong>Ошибка:</strong> {result.get('error', 'Неизвестная ошибка')}</p>
            <p><strong>Детали:</strong> {result.get('detail', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    overall_score = result.get("overall_score", 0)
    overall_quality = result.get("overall_quality", "")
    is_healthy = result.get("is_healthy", False)
    
    # Общая оценка
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Общая оценка", f"{overall_score:.1f}/5.0", f"{overall_quality}")
    
    with col2:
        health_emoji = "🟢" if is_healthy else "🔴"
        health_text = "Здоровое" if is_healthy else "Болезнь обнаружена"
        st.metric("Состояние здоровья", health_text, health_emoji)
    
    with col3:
        quality_data = result.get("quality", {})
        quality_class = quality_data.get("predicted_class_ru", "")
        st.metric("Качество", quality_class)
    
    # Детальная информация
    col1, col2 = st.columns(2)
    
    with col1:
        # Информация о болезнях
        disease_data = result.get("disease", {})
        disease_class = disease_data.get("predicted_class_ru", "")
        disease_confidence = disease_data.get("confidence", 0)
        
        st.markdown(f"""
        <div class="metric-container">
            <h4>🦠 Анализ болезней</h4>
            <p><strong>Состояние:</strong> {disease_class}</p>
            <p><strong>Уверенность:</strong> {disease_confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Информация о зрелости
        maturity_data = result.get("maturity", {})
        maturity_class = maturity_data.get("predicted_class_ru", "")
        maturity_confidence = maturity_data.get("confidence", 0)
        
        st.markdown(f"""
        <div class="metric-container">
            <h4>🌱 Стадия зрелости</h4>
            <p><strong>Стадия:</strong> {maturity_class}</p>
            <p><strong>Уверенность:</strong> {maturity_confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Рекомендации
    recommendations = result.get("recommendations", [])
    if recommendations:
        st.markdown("### 📋 Рекомендации")
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="recommendation-box">
                <strong>{i}.</strong> {rec}
            </div>
            """, unsafe_allow_html=True)

def display_yield_results(result: Dict):
    """Отображение результатов прогнозирования урожайности"""
    if not result.get("success", False):
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ Ошибка прогнозирования урожайности</h4>
            <p><strong>Ошибка:</strong> {result.get('error', 'Неизвестная ошибка')}</p>
            <p><strong>Детали:</strong> {result.get('detail', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    predicted_yield = result.get("predicted_yield_tons_per_ha", 0)
    confidence = result.get("confidence", 0)
    prediction_range = result.get("prediction_range", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Прогнозируемая урожайность", 
            f"{predicted_yield:.2f} т/га",
            f"Уверенность: {confidence:.1%}"
        )
    
    with col2:
        lower = prediction_range.get("lower", 0)
        upper = prediction_range.get("upper", 0)
        st.metric(
            "Диапазон прогноза", 
            f"{lower:.1f} - {upper:.1f} т/га",
            f"±{(upper-lower)/2:.1f} т/га"
        )
    
    with col3:
        # Качество прогноза
        if confidence >= 0.8:
            quality_text = "Высокое"
            quality_color = "🟢"
        elif confidence >= 0.6:
            quality_text = "Среднее"
            quality_color = "🟡"
        else:
            quality_text = "Низкое"
            quality_color = "🔴"
        
        st.metric("Качество прогноза", quality_text, quality_color)
    
    # График диапазона прогноза
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[lower, predicted_yield, upper],
        y=[1, 1, 1],
        mode='markers+lines',
        marker=dict(size=[8, 12, 8], color=['red', 'green', 'red']),
        line=dict(color='blue', width=2),
        name='Прогноз урожайности'
    ))
    
    fig.update_layout(
        title="Диапазон прогнозируемой урожайности",
        xaxis_title="Урожайность (т/га)",
        yaxis=dict(visible=False),
        height=200,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Рекомендации
    recommendations = result.get("recommendations", [])
    if recommendations:
        st.markdown("### 📋 Рекомендации по урожайности")
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="recommendation-box">
                <strong>{i}.</strong> {rec}
            </div>
            """, unsafe_allow_html=True)

def display_comprehensive_results(result: Dict):
    """Отображение результатов комплексного анализа"""
    if not result.get("success", False):
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ Ошибка комплексного анализа</h4>
            <p><strong>Ошибка:</strong> {result.get('error', 'Неизвестная ошибка')}</p>
            <p><strong>Детали:</strong> {result.get('detail', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Извлекаем результаты
    crop_result = result.get("crop_classification", {})
    quality_result = result.get("quality_assessment", {})
    yield_result = result.get("yield_prediction", {})
    comprehensive_recommendations = result.get("comprehensive_recommendations", [])
    
    # Сводная информация
    st.markdown("## 📊 Сводная информация")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        crop_class = crop_result.get("predicted_class_ru", "Неизвестно")
        st.metric("Культура", crop_class)
    
    with col2:
        overall_score = quality_result.get("overall_score", 0)
        st.metric("Качество", f"{overall_score:.1f}/5.0")
    
    with col3:
        predicted_yield = yield_result.get("predicted_yield_tons_per_ha", 0)
        st.metric("Урожайность", f"{predicted_yield:.1f} т/га")
    
    with col4:
        is_healthy = quality_result.get("is_healthy", False)
        health_status = "Здоровое" if is_healthy else "Требует внимания"
        health_emoji = "🟢" if is_healthy else "🔴"
        st.metric("Здоровье", health_status, health_emoji)
    
    # Детальные результаты в табах
    tab1, tab2, tab3 = st.tabs(["🌾 Классификация", "🔍 Качество", "📈 Урожайность"])
    
    with tab1:
        # Создаем фиктивный успешный результат для классификации
        classification_response = {
            "success": True,
            **crop_result
        }
        display_classification_results(classification_response)
    
    with tab2:
        # Создаем фиктивный успешный результат для качества
        quality_response = {
            "success": True,
            **quality_result
        }
        display_quality_results(quality_response)
    
    with tab3:
        # Создаем фиктивный успешный результат для урожайности
        yield_response = {
            "success": True,
            **yield_result
        }
        display_yield_results(yield_response)
    
    # Комплексные рекомендации
    if comprehensive_recommendations:
        st.markdown("## 🎯 Комплексные рекомендации")
        for i, rec in enumerate(comprehensive_recommendations, 1):
            st.markdown(f"""
            <div class="recommendation-box">
                <strong>{i}.</strong> {rec}
            </div>
            """, unsafe_allow_html=True)

def main():
    """Главная функция приложения"""
    # Заголовок
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Агропайплайн</h1>
        <h3>Интеллектуальный анализ сельскохозяйственных культур</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки анализа")
        
        analysis_type = st.selectbox(
            "Тип анализа",
            ["Комплексный анализ", "Классификация культуры", "Оценка качества", "Прогноз урожайности"],
            help="Выберите тип анализа для загруженного изображения"
        )
        
        st.markdown("---")
        
        st.markdown("""
        ### 📝 Инструкция
        1. Выберите тип анализа
        2. Загрузите изображение культуры
        3. Дождитесь результатов анализа
        4. Изучите рекомендации
        
        ### 📋 Поддерживаемые форматы
        - JPG, JPEG
        - PNG
        - BMP, TIFF
        
        ### 🎯 Поддерживаемые культуры
        - 🌾 Пшеница
        - 🌾 Ячмень  
        - 🌽 Кукуруза
        """)
    
    # Основной контент
    st.header("📤 Загрузка изображения")
    
    uploaded_file = st.file_uploader(
        "Выберите изображение культуры",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Загрузите качественное изображение сельскохозяйственной культуры"
    )
    
    if uploaded_file is not None:
        # Отображение загруженного изображения
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", use_column_width=True)
            
            st.markdown("### 📏 Информация об изображении")
            display_image_info(image)
        
        with col2:
            if st.button("🚀 Запустить анализ", type="primary"):
                with st.spinner("Анализ в процессе..."):
                    # Вызываем API
                    result = call_api(analysis_type, uploaded_file)
                    
                    # Отображаем результаты
                    st.markdown("---")
                    st.header(f"📊 Результаты: {analysis_type}")
                    
                    if analysis_type == "Комплексный анализ":
                        display_comprehensive_results(result)
                    elif analysis_type == "Классификация культуры":
                        display_classification_results(result)
                    elif analysis_type == "Оценка качества":
                        display_quality_results(result)
                    elif analysis_type == "Прогноз урожайности":
                        display_yield_results(result)
    
    else:
        st.info("👆 Загрузите изображение для начала анализа")
        
        # Демонстрационные примеры
        st.markdown("---")
        st.header("🎯 Примеры анализа")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="success-box">
                <h4>🌾 Классификация</h4>
                <p>Определение типа культуры с точностью >90%</p>
                <ul>
                    <li>Пшеница</li>
                    <li>Ячмень</li>
                    <li>Кукуруза</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="success-box">
                <h4>🔍 Оценка качества</h4>
                <p>Анализ состояния и здоровья растений</p>
                <ul>
                    <li>Болезни</li>
                    <li>Зрелость</li>
                    <li>Общее качество</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="success-box">
                <h4>📈 Прогноз урожайности</h4>
                <p>Предсказание урожайности в т/га</p>
                <ul>
                    <li>Плотность растений</li>
                    <li>Здоровье культуры</li>
                    <li>Стадия развития</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 