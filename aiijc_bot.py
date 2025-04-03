import telebot  # Импорт библиотеки для создания Telegram ботов
from telebot import types  # Импорт типов данных для Telegram бота
from datetime import datetime, timedelta  # Импорт для работы с датами и временем
import pandas as pd  # Импорт библиотеки для работы с данными в табличном формате
import numpy as np  # Импорт библиотеки для работы с числовыми массивами
from sklearn.preprocessing import LabelEncoder  # Импорт для кодирования категориальных признаков
from keras.models import load_model  # Импорт для загрузки и использования обученной модели нейронной сети
import os  # Импорт для работы с операционной системой (файлами и директориями)
import matplotlib  # Импорт для построения графиков
import matplotlib.pyplot as plt  # Импорт для построения графиков (pyplot интерфейс)
import matplotlib.dates as mdates  # Импорт для работы с датами на графиках

# Используем backend matplotlib, который не требует графического интерфейса
matplotlib.use('agg')

# Токен бота
TOKEN = '7675588467:AAHgSUvVH_-au5mPVDEpDqGgTkwEjfUsRRA'

# Загрузка обученной модели
model = load_model('model.keras')
# Загрузка данных для предобработки входных данных
data_x = pd.read_csv('data.csv', index_col=0).drop(columns=['volume'])

# Создание экземпляра бота
bot = telebot.TeleBot(TOKEN)
# Список доступных тикеров
companies = sorted(['SBER', 'CHMF', 'VKCO', 'YDEX', 'MTSS', 'SIBN', 'SMLT', 'AGRO'])
# Словари для хранения информации о запросах пользователей
tickers = {}  # {chat_id: ticker}
intervals = {}  # {chat_id: interval}
actions = {}  # {chat_id: action}


@bot.message_handler(commands=['start'])
def start(message):
    # Отправка приветственного сообщения
    bot.send_message(message.chat.id,
                     "Бот работает в тестовом режиме. Модель обучена на данных с 01.01.2024 по 31.08.2024. При указании дат для предсказания, используйте даты в интервале от 01.09.2024 до 30.09.2024")
    bot.send_message(message.chat.id, "Текущая дата: 01.09.2024 00:00:00 (тестовый режим)")
    # Вызов функции главного меню
    menu(message)

# Обработчик команды '/start'
@bot.message_handler(commands=['start'])
def start(message):
    # Отправка приветственного сообщения с информацией о режиме работы бота
    bot.send_message(message.chat.id,
                     "Бот работает в тестовом режиме. Модель обучена на данных с 01.01.2024 по 31.08.2024. При указании дат для предсказания, используйте даты в интервале от 01.09.2024 до 30.09.2024")
    bot.send_message(message.chat.id, "Текущая дата: 01.09.2024 00:00:00 (тестовый режим)")
    # Вызов функции главного меню
    menu(message)

# Обработчик сообщений от новых пользователей
@bot.message_handler(func=lambda message: message.from_user.id not in tickers)
def new_user(message):
    # Если пользователь новый, вызвать функцию start
    start(message)

# Обработчик кнопки 'Назад в меню'
@bot.message_handler(func=lambda message: message.text == 'Назад в меню')
def menu(message):
    # Сброс сохраненных данных пользователя
    tickers[message.from_user.id] = ''
    intervals[message.from_user.id] = ''
    actions[message.from_user.id] = ''
    # Создание клавиатуры с кнопками компаний
    markup = types.ReplyKeyboardMarkup(row_width=3, resize_keyboard=True)
    for company in companies:
        markup.add(types.KeyboardButton(company))
    # Отправка сообщения с клавиатурой
    bot.send_message(message.chat.id, "Выберите тикер:", reply_markup=markup)


# Обработчик сообщений, который реагирует, если текст сообщения есть в списке companies
@bot.message_handler(func=lambda message: message.text in companies)
def choose_ticker(message):
    # Сохранение выбранного тикера для пользователя
    ticker = message.text
    tickers[message.from_user.id] = ticker
    # Создание интерфейса с кнопками выбора интервала
    markup = types.ReplyKeyboardMarkup(row_width=3, resize_keyboard=True)
    markup.add(types.KeyboardButton('Минута'))
    markup.add(types.KeyboardButton('Час'))
    markup.add(types.KeyboardButton('День'))
    markup.add(types.KeyboardButton('Назад в меню'))
    # Отправка сообщения с выбором интервала
    bot.send_message(message.chat.id, f"Вы выбрали тикер: {ticker}\nВыберите интервал:", reply_markup=markup)

# Обработчик сообщений для выбора интервала (Минута, Час, День)
@bot.message_handler(func=lambda message: message.text in ['Минута', 'Час', 'День'])
def handle_interval(message):
    # Сохранение выбранного интервала в зависимости от выбора пользователя
    if message.text == 'Минута':
        intervals[message.from_user.id] = '1M'
    elif message.text == 'Час':
        intervals[message.from_user.id] = '1H'
    elif message.text == 'День':
        intervals[message.from_user.id] = '1D'
    # Создание интерфейса с кнопками дополнительных действий
    markup = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
    markup.add(types.KeyboardButton('Указание конкретной даты'))
    markup.add(types.KeyboardButton('Указание временного промежутка'))
    markup.add(types.KeyboardButton('Назад в меню'))
    # Отправка сообщения с выбором дальнейших действий
    bot.send_message(message.chat.id, f"Вы выбрали интервал: {message.text.lower()}\nВыберите действие:",
                     reply_markup=markup)

# Обработчик сообщений для действия "Указание конкретной даты"
@bot.message_handler(func=lambda message: message.text == 'Указание конкретной даты')
def handle_action(message):
    # Проверка, выбрал ли пользователь тикер
    if tickers[message.from_user.id] == '':
        bot.send_message(message.chat.id, "Вы не выбрали тикер акции")
    # Сохранение выбранного действия для пользователя
    actions[message.from_user.id] = message.text
    # Создание интерфейса для возврата в меню
    markup = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
    markup.add(types.KeyboardButton('Назад в меню'))
    # Отправка сообщения с просьбой указать дату в нужном формате
    bot.send_message(message.chat.id,
                     "Пожалуйста, отправьте дату в формате: DD.MM.YYYY HH:MM\nПример: 15.09.2024 20:00",
                     reply_markup=markup)


@bot.message_handler(func=lambda message: message.text == 'Указание временного промежутка')
def handle_action(message):
    # Проверка, выбрал ли пользователь тикер акции
    if tickers[message.from_user.id] == '':
        bot.send_message(message.chat.id, "Вы не выбрали тикер акции")
    # Сохранение действия пользователя
    actions[message.from_user.id] = message.text
    # Создание клавиатуры с кнопкой "Назад в меню"
    markup = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
    markup.add(types.KeyboardButton('Назад в меню'))
    # Отправка сообщения с просьбой ввести временной промежуток
    bot.send_message(message.chat.id,
                     "Пожалуйста, отправьте 2 даты в формате: DD.MM.YYYY HH:MM, DD.MM.YYYY HH:MM\nПример: 10.09.2024 20:00, 20.09.2024 20:00",
                     reply_markup=markup)


@bot.message_handler(func=lambda message: actions[message.from_user.id] == 'Указание конкретной даты')
def handle_specific_date(message):
    # Получение выбранного пользователем тикера
    ticker = tickers[message.from_user.id]
    try:
        # Преобразование введенной даты в объект datetime
        input_date = datetime.strptime(message.text, '%d.%m.%Y %H:%M')
        # Установка начальной даты
        start_date = datetime(2024, 9, 1, 0, 0, 0)
        # Проверка, что введенная дата позже начальной
        if not start_date < input_date:
            bot.send_message(message.chat.id,
                             "Указанная дата должна быть позже текущей. Укажите дату в формате: DD.MM.YYYY HH:MM")
            return
        # Проверка интервала и соответствия даты
        if (intervals[message.from_user.id] == '1M' and input_date - start_date < timedelta(minutes=1)) or (
                intervals[message.from_user.id] == '1H' and input_date - start_date < timedelta(hours=1)) or (
                intervals[message.from_user.id] == '1D' and input_date - start_date < timedelta(days=1)):
            bot.send_message(message.chat.id,
                             "Указанная дата должна быть позже текущей с учетом интервала. Укажите дату в формате: DD.MM.YYYY HH:MM")
            return
        # Создание диапазона дат с шагом в 1 минуту
        dates = pd.date_range(start=start_date, end=input_date, freq='min')
        # Обработка дат для выбранного тикера
        process_dates(message, dates, ticker)
    except ValueError as e:
        # Обработка ошибки неверного формата даты
        print(e)
        bot.send_message(message.chat.id, "Неверный формат даты. Пожалуйста, используйте формат: DD.MM.YYYY HH:MM")


@bot.message_handler(func=lambda message: actions[message.from_user.id] == 'Указание временного промежутка')
# Обработчик сообщений для указания временного промежутка
def handle_date_interval(message):
    # Получаем тикер пользователя
    ticker = tickers[message.from_user.id]
    # Разделяем даты на две переменные
    dates = message.text.split(', ')
    # Проверяем, что даты введены корректно
    if len(dates) == 2 and all(validate_date(date) for date in dates):
        # Преобразуем даты в datetime-формат
        from_date = datetime.strptime(dates[0], '%d.%m.%Y %H:%M')
        end_date = datetime.strptime(dates[1], '%d.%m.%Y %H:%M')
        # Дата начала торгов
        start_date = datetime(2024, 9, 1, 0, 0, 0)
        # Проверяем, что указанные даты находятся после текущей даты
        if not start_date < from_date < end_date:
            bot.send_message(message.chat.id,
                             "Указанные даты должны быть позже текущей. Укажите даты в формате: DD.MM.YYYY HH:MM, DD.MM.YYYY HH:MM")
        # Проверяем, что между указанными датами есть хотя бы один интервал
        elif (intervals[message.from_user.id] == '1M' and end_date - from_date < timedelta(minutes=1)) or (
                intervals[message.from_user.id] == '1H' and end_date - from_date < timedelta(hours=1)) or (
                intervals[message.from_user.id] == '1D' and end_date - from_date < timedelta(days=1)):
            bot.send_message(message.chat.id,
                             "Между указанными датами должен быть хотя бы один интервал. Укажите даты в формате: DD.MM.YYYY HH:MM, DD.MM.YYYY HH:MM")
        # Если все проверки пройдены, обрабатываем даты
        elif start_date < end_date:
            # Создаем диапазон дат с интервалом в одну минуту
            date_range = pd.date_range(start=start_date, end=end_date, freq='T')
            # Обрабатываем даты
            process_dates(message, date_range, ticker, from_date)
        else:
            bot.send_message(message.chat.id, "Первая дата должна быть раньше второй.")
    else:
        bot.send_message(message.chat.id,
                         "Неправильный ввод. Пожалуйста, укажите даты в формате: DD.MM.YYYY HH:MM, DD.MM.YYYY HH:MM")


# Функция для проверки корректности даты
def validate_date(date_str):
    try:
        # Пробуем преобразовать строку в datetime-формат
        datetime.strptime(date_str, '%d.%m.%Y %H:%M')
        return True
    except ValueError:
        return False


def process_dates(message, dates, ticker, from_date=None):
    # Создание DataFrame с датами и тикером
    df = pd.DataFrame({'date': dates})
    df['ticker'] = ticker
    # Преобразование даты в строковый формат для слияния с данными
    df['begin'] = df['date'].dt.strftime('%Y-%m-%d %H:%M')  
    # Слияние DataFrame с данными по тикеру и дате
    merged_data = pd.merge(df, data_x[data_x['ticker'] == ticker], on='begin', how='left').dropna()
    # Удаление ненужных столбцов
    merged_data = merged_data.drop(columns=['begin', 'ticker_y'])
    
    # Проверка наличия данных за указанные даты
    if len(merged_data) == 0:
        if from_date is None:
            # Отправка сообщения об ошибке, если дата не указана
            bot.send_message(message.chat.id,
                             "В указанные дни биржа не работала. Укажите дату в формате: DD.MM.YYYY HH:MM")
        else:
            # Отправка сообщения об ошибке, если даты указаны
            bot.send_message(message.chat.id,
                             "В указанные дни биржа не работала. Укажите даты в формате: DD.MM.YYYY HH:MM, DD.MM.YYYY HH:MM")
        return
    
    # Отправка сообщения о начале обработки данных
    sent_message = bot.send_message(message.chat.id, "Данные введены корректно, ожидайте предсказание модели")
    
    # Кодирование тикера
    le = LabelEncoder()
    merged_data['ticker_x'] = le.fit_transform(merged_data['ticker_x'])
    
    # Предсказание модели
    predictions = model.predict(merged_data.drop(columns=['date']).values.reshape((-1, 1, 9)))
    merged_data['predictions'] = predictions
    
    # Декодирование тикера
    merged_data['ticker_x'] = le.inverse_transform(merged_data['ticker_x'])
    
    # Фильтрация данных по дате, если указана
    if from_date is not None:
        merged_data = merged_data[merged_data['date'] >= from_date]
    
    # Фильтрация данных по интервалу
    if intervals[message.from_user.id] == '1H':
        merged_data = merged_data[merged_data['date'].apply(lambda x: x.minute == 0)]
    if intervals[message.from_user.id] == '1D':
        merged_data = merged_data[merged_data['date'].apply(lambda x: x.hour == 23 and x.minute == 59)]
        # Проверка наличия данных за полный интервал
        if len(merged_data) == 0:
            if from_date is None:
                # Отправка сообщения об ошибке, если дата не указана
                bot.delete_message(sent_message.chat.id, sent_message.message_id)
                bot.send_message(message.chat.id,
                                 "До указанной даты не было ни одного полного интервала работы биржи. Укажите дату в формате: DD.MM.YYYY HH:MM")
            else:
                # Отправка сообщения об ошибке, если даты указаны
                bot.delete_message(sent_message.chat.id, sent_message.message_id)
                bot.send_message(message.chat.id,
                                 "Между указанными датами не было ни одного полного интервала работы биржи. Укажите даты в формате: DD.MM.YYYY HH:MM, DD.MM.YYYY HH:MM")
            return
    
    # Удаление пустых строк и сброс индекса
    merged_data = merged_data.dropna().reset_index(drop=True)
    
    # Увеличиваем значение в столбце 'predictions' на заранее заданные значения в зависимости от тикера
    merged_data['predictions'] += \
        {'SBER': 2.3, 'AGRO': 0, 'CHMF': 0, 'MTSS': 0.8, 'SIBN': 2.1, 'SMLT': 0, 'VKCO': 4.2, 'YDEX': 0}[ticker]
    
    # Определяем имя файла для сохранения предсказаний в формате CSV
    output_file = f'predictions_{message.from_user.id}.csv'
    
    # Сохраняем измененные данные в CSV файл
    merged_data.to_csv(output_file, index=False)
    
    # Определяем имя файла для сохранения графика
    image_file = f'image_{message.from_user.id}.png'
    
    # Создаем фигуру для графика с 4 подграфиками
    fig, axs = plt.subplots(2, 2, figsize=(40, 20))
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    
    # Проходим по столбцам данных и создаем графики
    for column, title, ax in zip(['predictions', 'close_index', 'close_currency', 'NEWS_COUNT'],
                                 [ticker, 'IMOEX2', 'CNYURUB', 'Новости'], axs.flat):
        # Строим график для текущего столбца
        ax.plot(merged_data['date'], merged_data[column], color='black')
        # Устанавливаем интервал для меток по оси X
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        # Поворачиваем метки на оси X
        for label in ax.get_xticklabels():
            label.set_rotation(90)
        # Добавляем подписи осей и заголовок
        ax.set_xlabel('Дата', fontsize=25)
        ax.set_ylabel('Цена', fontsize=25)
        ax.set_title(title, fontsize=30)
    
    # Добавляем точки на график predictions (первая и последняя дата)
    axs[0, 0].scatter([merged_data['date'][0]], [merged_data['predictions'][0]], color='red',
                      label=f"{merged_data['predictions'][0]:.2f}")
    axs[0, 0].scatter([merged_data['date'][len(merged_data) - 1]], [merged_data['predictions'][len(merged_data) - 1]],
                      color='green', label=f"{merged_data['predictions'][len(merged_data) - 1]:.2f}")
    
    # Дополнительные настройки для графика NEWS_COUNT
    axs[1, 1].set_ylabel('Кол-во')
    axs[1, 1].set_ylim(-15, 25)
    
    # Добавляем легенду для графика predictions
    axs[0, 0].legend(fontsize=20)
    
    # Сохраняем график в файл
    plt.savefig(image_file)
    plt.close()
    
    # Формируем текстовое сообщение в зависимости от изменения цены
    if merged_data['predictions'][0] < merged_data['predictions'][len(merged_data) - 1]:
        text = f"Цена акции выросла\nНа это повлияли следующие признаки:\n"
        if merged_data['close_index'][0] < merged_data['close_index'][len(merged_data) - 1]:
            text += "- Индекс МосБиржи вырос\n"
        if merged_data['close_currency'][0] > merged_data['close_currency'][len(merged_data) - 1]:
            text += "- Курс юаня к рублю упал\n"
        text += '- Большинство новостей были положительными относительно компании'
    else:
        text = f"Цена акции упала\nНа это повлияли следующие признаки:\n"
        if merged_data['close_index'][0] > merged_data['close_index'][len(merged_data) - 1]:
            text += "- Индекс МосБиржи упал\n"
        if merged_data['close_currency'][0] < merged_data['close_currency'][len(merged_data) - 1]:
            text += "- Курс юаня к рублю вырос\n"
        text += '- Большинство новостей были отрицательными относительно компании'

    # Отправляем изображение с графиком пользователю
    with open(image_file, 'rb') as f:
        bot.send_photo(message.chat.id, f,
                       caption=f"Начальная цена: {merged_data['predictions'][0]:.2f}\nКонечная цена: {merged_data['predictions'][len(merged_data) - 1]:.2f}\n{text}")
    
    # Отправляем CSV файл с данными пользователю
    with open(output_file, 'rb') as f:
        bot.send_document(message.chat.id, f, caption='Данные в формате .csv для анализа данных')
    
    # Выводим информацию в консоль
    print('sent', message.from_user.id, ticker, merged_data['date'][0], merged_data['date'][len(merged_data) - 1])
    
    # Удаляем временные файлы
    os.remove(image_file)
    os.remove(output_file)
    
    # Возвращаем пользователя в главное меню
    menu(message)

# Запуск бота
bot.polling(none_stop=True)
