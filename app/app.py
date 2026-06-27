# Streamlit — веб-интерфейс для анализа вторичного рынка жилья Москвы

import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from _utils import load_data, add_transformed_columns

# Настройки страницы и стиля графиков
st.set_page_config(page_title='Housing Market Analysis', layout='wide')
sns.set_style('whitegrid')

# Загружаем данные и добавляем трансформированные колонки
df = load_data()
df_t = add_transformed_columns(df)

st.title('Moscow Secondary Housing Market Analysis')

# --- 0. Введение ---
st.header('0. Introduction')
st.markdown('''
> This project aims to provide a clear perspective on the Moscow secondary market nowadays
> through comprehensive research based on the latest statistics (from 2020-2026) for both
> potential personal and commercial use. It was fully done by HSE 1st year undergraduate
> student Filippovskiy Denis.
''')

# --- 1. Обзор данных ---
st.header('1. Data Overview')
st.write(f'Dataset contains **{df.shape[0]}** rows and **{df.shape[1]}** columns.')

col1, col2 = st.columns(2)
with col1:
    st.subheader('First rows')
    st.dataframe(df.head())
with col2:
    st.subheader('Data types')
    st.write(df.dtypes.astype(str))

st.subheader('Descriptive Statistics')
st.dataframe(df.describe())

# --- 2. Очистка данных ---
st.header('2. Data Cleanup')
missing = df.isnull().sum()
if missing.sum() == 0:
    st.success('No missing values found in the dataset.')
else:
    st.warning(f'Missing values: {missing[missing > 0].to_dict()}')

# --- 3. Исследование данных ---
st.header('3. Data Exploration')

# Вкладки с разными графиками
tab1, tab2, tab3, tab4 = st.tabs(['Distributions', 'Price by Okrug', 'Correlation Matrix', 'Scatter Plots'])

# Вкладка 1: гистограммы распределений
with tab1:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(df['price_rub'] / 1e6, bins=50, edgecolor='black')
    axes[0, 0].set_xlabel('Price, millions RUB')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Apartment Prices')
    axes[0, 1].hist(df['total_area'], bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('Total Area, sqm')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Total Area')
    axes[1, 0].hist(df['price_per_sqm'] / 1e3, bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('Price per sqm, thousands RUB')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Distribution of Price per sqm')
    axes[1, 1].hist(df['to_center_km'], bins=50, edgecolor='black')
    axes[1, 1].set_xlabel('Distance to Center, km')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Distribution of Distance to Center')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Вкладка 2: средняя цена по округам
with tab2:
    okrug_price = df.groupby('okrug')['price_rub'].agg(['mean', 'median']).sort_values('mean')
    okrug_price['mean'] = okrug_price['mean'] / 1e6
    okrug_price['median'] = okrug_price['median'] / 1e6
    fig, ax = plt.subplots(figsize=(12, 6))
    okrug_price.plot(kind='bar', ax=ax)
    ax.set_ylabel('Price, millions RUB')
    ax.set_title('Average and Median Price by Okrug')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)
    plt.close()

# Вкладка 3: матрица корреляции
with tab3:
    num_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)
    plt.close()

# Вкладка 4: точечные графики зависимости цены от разных факторов
with tab4:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(df['total_area'], df['price_rub'] / 1e6, alpha=0.3, s=1)
    axes[0].set_xlabel('Total Area, sqm')
    axes[0].set_ylabel('Price, millions RUB')
    axes[0].set_title('Price vs Total Area')
    axes[1].scatter(df['building_year'], df['price_per_sqm'] / 1e3, alpha=0.3, s=1)
    axes[1].set_xlabel('Building Year')
    axes[1].set_ylabel('Price per sqm, thousands RUB')
    axes[1].set_title('Price per sqm vs Building Year')
    axes[2].scatter(df['to_center_km'], df['price_rub'] / 1e6, alpha=0.3, s=1)
    axes[2].set_xlabel('Distance to Center, km')
    axes[2].set_ylabel('Price, millions RUB')
    axes[2].set_title('Price vs Distance to Center')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# --- 4. Проверка гипотез ---
st.header('4. Hypothesis Testing')

# H1: зависимость цены от площади
st.subheader('H1: Larger area -> higher price')
corr1 = df['total_area'].corr(df['price_rub'])
st.write(f'Correlation: **{corr1:.3f}**. There is a strong positive relationship between area and price.')

# H2: зависимость цены за м² от года постройки
st.subheader('H2: Newer buildings -> higher price per sqm')
corr2 = df['building_year'].corr(df['price_per_sqm'])
st.write(f'Correlation: **{corr2:.3f}**. Building year alone is not a strong predictor of price per sqm.')

# H3: зависимость цены от расстояния до центра
st.subheader('H3: Closer to center -> higher price')
corr3 = df['to_center_km'].corr(df['price_rub'])
st.write(f'Correlation: **{corr3:.3f}**. Apartments closer to the center tend to be more expensive.')

# H4: сравнение цен агентств и собственников
st.subheader('H4: Agency vs Owner prices')
agency_mean = df[df['seller_type'] == 'agency']['price_rub'].mean() / 1e6
owner_mean = df[df['seller_type'] == 'owner']['price_rub'].mean() / 1e6

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
agency_prices = df[df['seller_type'] == 'agency']['price_rub'] / 1e6
owner_prices = df[df['seller_type'] == 'owner']['price_rub'] / 1e6
axes[0].hist(agency_prices, bins=50, alpha=0.7, label='Agency', edgecolor='black')
axes[0].hist(owner_prices, bins=50, alpha=0.7, label='Owner', edgecolor='black')
axes[0].set_xlabel('Price, millions RUB')
axes[0].set_ylabel('Count')
axes[0].set_title('Price Distribution by Seller Type')
axes[0].legend()
axes[1].bar(['Agency', 'Owner'], [agency_mean, owner_mean], edgecolor='black')
axes[1].set_ylabel('Mean Price, millions RUB')
axes[1].set_title('Mean Price by Seller Type')
st.pyplot(fig)
plt.close()
st.write(f'Agency mean: **{agency_mean:.2f}M** RUB, Owner mean: **{owner_mean:.2f}M** RUB')

# --- 5. Трансформация данных ---
st.header('5. Data Transformation')
st.write('Added two new columns:')
st.code('''df['price_per_room'] = df['price_rub'] / df['rooms']
df['listing_year'] = pd.to_datetime(df['date_posted']).dt.year
df['building_age'] = df['listing_year'] - df['building_year']''')

st.dataframe(df_t[['id', 'total_area', 'rooms', 'price_rub', 'price_per_room', 'building_year', 'listing_year', 'building_age']].head())

# --- 6. Итоги ---
st.header('6. Summary')
st.markdown('''
- Dataset contains 50,000 Moscow secondary housing listings
- No missing values or data quality issues
- **Total area** is the strongest predictor of price (correlation ~0.67)
- **Distance to center** has moderate negative correlation with price (-0.28)
- **Agency listings** are on average more expensive than owner listings
- New columns: `price_per_room` and `building_age`
''')
