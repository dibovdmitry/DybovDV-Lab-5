# Dockerfile
# Используем официальный Python образ с нужной версией
FROM python:3.10-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app
 
# Копируем файл с зависимостями в первую очередь (для кэширования слоя)
COPY requirements.txt .
 
# Обновляем pip и устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip && \ 
    pip install --no-cache-dir -r requirements.txt
 
# Копируем весь исходный код проекта в контейнер
COPY . .
 
# Создаем непривилегированного пользователя для безопасности
RUN useradd -m -u 1000 user
USER user
 
# Сообщаем Docker, что контейнер слушает на порту 8000
EXPOSE 8000
 
# Команда для запуска приложения с Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
