# -*- coding: utf-8 -*-
import telebot
from telebot import types
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from PIL import Image
import io
import model

with open('TOKEN.txt', 'r') as f:
    token = f.readline()
bot = telebot.TeleBot(token)

def build_menu(buttons, n_cols,
               header_buttons=None,
               footer_buttons=None):
    '''Функция создания встроенного стартового меню'''
    menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
    if header_buttons:
        menu.insert(0, [header_buttons])
    if footer_buttons:
        menu.append([footer_buttons])
    return menu

@bot.message_handler(commands=['start'])
def welcome_start(msg, text='Dogs Breeds Classifier приветствует тебя!'):
    '''Функция Старт'''
    button_list = [InlineKeyboardButton('Классифицировать фото', callback_data='foto')]

    # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=1))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text=text, reply_markup=reply_markup)


@bot.callback_query_handler(func=lambda c: c.data == 'foto')
def button_foto(callback_query: types.CallbackQuery):
    '''Функция определения ответа на кнопку'''
    bot.answer_callback_query(callback_query.id)
    bot.send_message(callback_query.from_user.id, 'Загрузите фото!')

@bot.message_handler(content_types=['photo'])
def image_open(msg):
    if (msg.text != 'Назад'):
        bot.send_message(msg.chat.id, "Секунду, это...")
        try:
            file_info = bot.get_file(msg.photo[len(msg.photo) - 1].file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            file_path = io.BytesIO(downloaded_file) # декодируем изображение из байт
            # предсказание класса
            with Image.open(file_path) as im:
                label = model.predict_image(im)
                bot.send_message(msg.chat.id, label.capitalize())

        except Exception as e:
            pass
    else:
        bot.send_message(msg, "Хорошо, отменяем. Чем еще могу помочь?")


bot.polling(none_stop=True)