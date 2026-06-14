import pandas as pd
import os

# Путь к файлу с данными (относительно корня проекта)
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'secondary_market.csv')


# Загружает датасет из CSV
def load_data():
    return pd.read_csv(DATA_PATH)


# Возвращает базовую статистику по числовым колонкам
def get_basic_stats(df):
    return df.describe()


# Возвращает количество пропущенных значений в каждой колонке
def get_missing_values(df):
    return df.isnull().sum()


# Добавляет две новые колонки: price_per_room и building_age
def add_transformed_columns(df):
    df = df.copy()
    # Цена за одну комнату
    df['price_per_room'] = df['price_rub'] / df['rooms']
    # Год публикации объявления
    df['listing_year'] = pd.to_datetime(df['date_posted']).dt.year
    # Возраст здания на момент публикации
    df['building_age'] = df['listing_year'] - df['building_year']
    return df


# Фильтрует данные по параметрам и возвращает с пагинацией
def filter_data(df, okrug=None, seller_type=None, min_price=None, max_price=None, limit=100, offset=0):
    result = df.copy()
    if okrug:
        result = result[result['okrug'] == okrug]
    if seller_type:
        result = result[result['seller_type'] == seller_type]
    if min_price is not None:
        result = result[result['price_rub'] >= min_price]
    if max_price is not None:
        result = result[result['price_rub'] <= max_price]
    total = len(result)
    result = result.iloc[offset:offset + limit]
    return result, total
