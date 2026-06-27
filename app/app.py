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
st.write(f'Dataset contains **{df.shape[0]}** rows and **{df.shape[1]}** columns, yet no missing values, inconsistent values, wrong data types, etc were found in the dataset during the research.')

col_desc = pd.DataFrame([
    ('id', 'Unique listing identifier'),
    ('date_posted', 'Listing publication date'),
    ('district', 'Moscow district'),
    ('okrug', 'Administrative okrug (CAO, VAO, SAO, etc.)'),
    ('lat, lon', 'Apartment coordinates'),
    ('total_area', 'Total area, m²'),
    ('living_area', 'Living area, m²'),
    ('kitchen_area', 'Kitchen area, m²'),
    ('rooms', 'Number of rooms'),
    ('floor', 'Floor number'),
    ('total_floors', 'Total floors in the building'),
    ('building_year', 'Year of construction'),
    ('building_type', 'Building type (khrushchev, monolith, brick, etc.)'),
    ('ceiling_height', 'Ceiling height, m'),
    ('has_balcony', 'Balcony availability'),
    ('renovation', 'Renovation type (cosmetic, euro, designer, no_renovation)'),
    ('metro_station', 'Nearest metro station'),
    ('metro_line', 'Metro line name'),
    ('metro_distance_min', 'Distance to metro in minutes'),
    ('metro_distance_type', 'Distance type (walk / transport)'),
    ('to_center_km', 'Distance to Moscow center, km'),
    ('price_rub', 'Price in rubles'),
    ('price_per_sqm', 'Price per square meter'),
    ('mortgage_rate_at_listing', 'Mortgage rate at the time of listing'),
    ('seller_type', 'Seller type (agency / owner)'),
], columns=['Column', 'Description'])

st.subheader('Data description')
st.dataframe(col_desc)

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

# H1: Metro walk paradox
st.subheader('H1: Apartments 5–12 min walk from metro cost more per sqm than those within 3 min')
walk_df = df[df['metro_distance_type'] == 'walk'].copy()
walk_df['walk_group'] = pd.cut(walk_df['metro_distance_min'], bins=[0, 3, 5, 12, 200], labels=['0-3 min', '3-5 min', '5-12 min', '12+ min'])
avg_pps = walk_df.groupby('walk_group', observed=True)['price_per_sqm'].mean()
st.write(f'Mean price per sqm:')
st.dataframe(avg_pps.round(0).to_frame('avg price_per_sqm'))
nearby = avg_pps.loc['0-3 min'] if '0-3 min' in avg_pps.index else 0
mid = avg_pps.loc['5-12 min'] if '5-12 min' in avg_pps.index else 0
diff_pct = ((mid - nearby) / nearby) * 100
st.write(f'Mean price per sqm for 0–3 min walk: **{nearby:,.0f} ₽**. For 5–12 min walk: **{mid:,.0f} ₽** (+{diff_pct:.1f}%). Because being right on top of the metro results in constant noise, crowds and traffic around your apartment.')

# H2: зависимость цены за м² от года постройки
st.subheader('H2: Newer buildings -> higher price per sqm')
corr2 = df['building_year'].corr(df['price_per_sqm'])
st.write(f'Correlation: **{corr2:.3f}**. Building year alone is not a strong predictor of price per sqm.')

# H3: Balcony price premium
st.subheader('H3: Apartments with a balcony → higher price per sqm')
balcony_yes = df[df['has_balcony'] == True]['price_per_sqm'].mean()
balcony_no = df[df['has_balcony'] == False]['price_per_sqm'].mean()
premium = (balcony_yes - balcony_no) / balcony_no * 100
st.write(f'With balcony: **{balcony_yes:,.0f} ₽/м²**, without: **{balcony_no:,.0f} ₽/м²** (+{premium:.1f}%). Balconies add functional space and are valued in the Moscow market.')

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
- **Metro walk paradox**: apartments 5–12 min walk from metro cost more per sqm than those within 3 min
- **Distance to center** has moderate negative correlation with price (-0.28)
- **Agency listings** are on average more expensive than owner listings
- New columns: `price_per_room` and `building_age`
''')
