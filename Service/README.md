# Классификация пород собак
- сервис написан на Flask
- модель поддерживает 10 пород собак, <a href="https://github.com/fastai/imagenette"> источник данных </a>


## Структура проекта
- app.py - сервис на Flask
- model.py - модель классификации
- config.py - конфигурационный файл

**static**  
    img - каталог для сохранения картинок
- model_2.pt - сохраненная модель pytorch (densenet161)
- model_4.pt - сохраненная модель pytorch (densenet201)

**templates**
- home.html - шаблон стартовой страницы
- predict.html - шаблон рабочей страницы

## Методы
- POST /predict - предсказываем породу собаки

## Запуск проекта

- Выполните команду python app.py --host 0.0.0.0 --port 5000