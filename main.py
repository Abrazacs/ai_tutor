import uvicorn
from src.api.routes import app

if __name__ == "__main__":

    print("=" * 50)
    print("\nAI TUTOR API\n")
    print("=" * 50)
    print("\nЗапуск сервера...\n")
    print("Документация доступна по адресу: http://localhost:8000/docs\n")
    print("=" * 50)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
