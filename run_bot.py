import sys
from src.bot.simple_bot import SimpleTelegramBot


def main():
    """Главная функция запуска бота"""
    print("=" * 50)
    print("\n AI TUTOR - TELEGRAM BOT (ПРОСТАЯ ВЕРСИЯ)\n")
    print("=" * 50)
    print("\n Инициализация бота...\n")

    try:
        bot = SimpleTelegramBot()
        print("Бот инициализирован успешно")
        bot.run()

    except KeyboardInterrupt:
        print("\n\n Бот остановлен пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n Ошибка запуска бота: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()