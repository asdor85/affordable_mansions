# Telegram бот для просмотра данных о вторичном рынке жилья

import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import pandas as pd
import numpy as np

from _utils import load_data, add_transformed_columns

# Загружаем данные
df = load_data()
df_t = add_transformed_columns(df)

# Вставьте сюда токен вашего бота
TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'


# Команда /start — показывает меню с кнопками
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton('Data Overview', callback_data='overview')],
        [InlineKeyboardButton('Statistics', callback_data='stats')],
        [InlineKeyboardButton('Hypotheses', callback_data='hypotheses')],
        [InlineKeyboardButton('Sample Listings', callback_data='listings')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        'Welcome to Housing Market Analysis Bot!\nChoose a section:',
        reply_markup=reply_markup
    )


# Обработчик нажатий на кнопки меню
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    # Обзор датасета
    if data == 'overview':
        text = (
            f'Dataset: Moscow Secondary Housing Market\n'
            f'Rows: {df.shape[0]}\n'
            f'Columns: {df.shape[1]}\n\n'
            f'Columns:\n{", ".join(df.columns[:13])}\n{", ".join(df.columns[13:])}'
        )

    # Статистика по числовым колонкам
    elif data == 'stats':
        desc = df.describe()
        text = 'Numeric columns statistics:\n\n'
        for col in ['price_rub', 'total_area', 'living_area', 'price_per_sqm']:
            row = desc[col]
            text += f'{col}:\n  mean={row["mean"]:.0f}, std={row["std"]:.0f}\n  min={row["min"]:.0f}, max={row["max"]:.0f}\n\n'

    # Результаты проверки гипотез
    elif data == 'hypotheses':
        corr1 = df['total_area'].corr(df['price_rub'])
        corr2 = df['building_year'].corr(df['price_per_sqm'])
        corr3 = df['to_center_km'].corr(df['price_rub'])
        agency_mean = df[df['seller_type'] == 'agency']['price_rub'].mean() / 1e6
        owner_mean = df[df['seller_type'] == 'owner']['price_rub'].mean() / 1e6
        text = (
            f'H1: Area vs Price - corr={corr1:.3f}\n'
            f'H2: Year vs Price/sqm - corr={corr2:.3f}\n'
            f'H3: Distance to center vs Price - corr={corr3:.3f}\n'
            f'H4: Agency mean={agency_mean:.2f}M, Owner mean={owner_mean:.2f}M RUB'
        )

    # Примеры объявлений
    elif data == 'listings':
        sample = df.head(5)[['id', 'total_area', 'rooms', 'price_rub', 'okrug']].to_string(index=False)
        text = f'Sample listings:\n\n{sample}'

    else:
        text = 'Unknown option'

    await query.edit_message_text(text)


# Запуск бота
async def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CallbackQueryHandler(button_handler))
    print('Bot started...')
    await app.run_polling()

if __name__ == '__main__':
    asyncio.run(main())
