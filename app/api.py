# FastAPI — REST API для работы с данными вторичного рынка жилья

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from _utils import load_data, add_transformed_columns

# Создаём экземпляр FastAPI
app = FastAPI(title='Secondary Housing Market API')

# Загружаем данные при старте
df = load_data()


# Модель для создания нового объявления (POST)
class ListingCreate(BaseModel):
    total_area: float
    living_area: float
    kitchen_area: float
    rooms: int
    floor: int
    total_floors: int
    building_year: int
    ceiling_height: float
    has_balcony: bool
    metro_distance_min: int
    to_center_km: float
    price_rub: int
    seller_type: str


# GET /listings — получение объявлений с фильтрацией и пагинацией
@app.get('/listings')
def get_listings(
    okrug: str | None = None,
    seller_type: str | None = None,
    min_price: int | None = None,
    max_price: int | None = None,
    limit: int = 100,
    offset: int = 0
):
    result = df.copy()
    # Фильтр по округу
    if okrug:
        result = result[result['okrug'] == okrug]
    # Фильтр по типу продавца
    if seller_type:
        result = result[result['seller_type'] == seller_type]
    # Фильтр по минимальной цене
    if min_price is not None:
        result = result[result['price_rub'] >= min_price]
    # Фильтр по максимальной цене
    if max_price is not None:
        result = result[result['price_rub'] <= max_price]
    total = len(result)
    # Пагинация
    result = result.iloc[offset:offset + limit]
    return {
        'total': total,
        'limit': limit,
        'offset': offset,
        'data': result.to_dict(orient='records')
    }


# GET /stats — общая статистика по датасету
@app.get('/stats')
def get_stats():
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr().to_dict()
    desc = df.describe().to_dict()
    return {
        'shape': list(df.shape),
        'columns': list(df.columns),
        'missing': df.isnull().sum().to_dict(),
        'describe': desc,
        'correlation': corr
    }


# POST /listings — создание нового объявления
@app.post('/listings')
def create_listing(item: ListingCreate):
    global df
    # Генерируем новый ID
    new_id = f'S{len(df) + 1:06d}'
    new_row = {
        'id': new_id,
        'date_posted': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'district': 'Unknown',
        'okrug': 'Unknown',
        'lat': 0.0,
        'lon': 0.0,
        'total_area': item.total_area,
        'living_area': item.living_area,
        'kitchen_area': item.kitchen_area,
        'rooms': item.rooms,
        'floor': item.floor,
        'total_floors': item.total_floors,
        'building_year': item.building_year,
        'building_type': 'unknown',
        'ceiling_height': item.ceiling_height,
        'has_balcony': item.has_balcony,
        'renovation': 'unknown',
        'metro_station': 'Unknown',
        'metro_line': 'Unknown',
        'metro_distance_min': item.metro_distance_min,
        'metro_distance_type': 'transport',
        'to_center_km': item.to_center_km,
        'price_rub': item.price_rub,
        'price_per_sqm': int(item.price_rub / item.total_area),
        'mortgage_rate_at_listing': 0.0,
        'seller_type': item.seller_type
    }
    # Добавляем новую строку в датафрейм
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return {'message': 'Listing created', 'id': new_id}
