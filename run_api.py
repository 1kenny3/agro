#!/usr/bin/env python3
"""
Скрипт запуска API сервера для Агропайплайна
"""

import uvicorn
from src.config.settings import settings

def main():
    """Запуск FastAPI сервера"""
    print(f"🚀 Запуск {settings.PROJECT_NAME} v{settings.VERSION}")
    print(f"📡 API будет доступен по адресу: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"📖 Документация: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print(f"💻 Устройство: {settings.DEVICE}")
    print("-" * 50)
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else 4
    )

if __name__ == "__main__":
    main() 