#!/usr/bin/env python3
"""
Скрипт запуска Streamlit веб-интерфейса для Агропайплайна
"""

import subprocess
import sys
import os

def main():
    """Запуск Streamlit приложения"""
    print("🌾 Запуск Агропайплайн - Веб-интерфейс")
    print("🌐 Веб-интерфейс будет доступен по адресу: http://localhost:8501")
    print("⚙️ Убедитесь, что API сервер запущен на localhost:8000")
    print("-" * 50)
    
    # Запускаем Streamlit приложение
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "frontend/app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Веб-интерфейс остановлен")
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка запуска: {e}")
    except FileNotFoundError:
        print("❌ Streamlit не найден. Установите зависимости: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 