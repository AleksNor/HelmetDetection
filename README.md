# Разработка системы контроля ношения средств индивидуальной защиты (касок)

**HelmetDetection** — FastAPI-сервис для детекции людей и средств индивидуальной защиты (каски) на видеопотоке. Поддерживает локальные файлы, RTSP/HTTP потоки и веб-камеры; инференс выполняется на CPU или GPU (PyTorch / Ultralytics YOLO). Сохраняет снимки нарушений в `violations/`, отдаёт MJPEG-стрим для просмотра в браузере и простую страницу с просмотром зарегистрированных нарушений.

---

## Структура репозитория
```
.
├─ app/
│  ├─ main.py            # FastAPI, маршруты, MJPEG-стрим
│  ├─ processor.py       # VideoProcessor: capture, модель, логика нарушений
│  ├─ tracker.py         # простой IoU-трекер
│  └─ config.py          # pydantic-settings (Settings)
├─ runs/                 # модель: runs/detect/.../weights/best.pt
├─ violations/           # сохраняемые кадры (bind-mountable)
├─ Dockerfile.cpu
├─ Dockerfile.gpu
├─ docker-compose.yml
├─ requirements.txt
├─ .env                  # конфигурация окружения
└─ README.md
```

---

## Docker — CPU и GPU

### Предварительные требования для GPU
- Установлен NVIDIA драйвер и NVIDIA Container Toolkit.  
- Проверка: `nvidia-smi` и `docker info | grep -i nvidia`.

### Пример `.env`
```dotenv
VIDEO_SOURCE=http://192.168.1.230:8080/video
DEVICE=cuda               # "cuda" или "cpu"
PROCESS_FPS=25
VIOLATION_SECONDS=3.0
COOLDOWN_SECONDS=5.0
OUTPUT_DIR=/app/violations
MODEL_PATH=/app/runs/detect/sh17_person_head_helmet/weights/best.pt
```

### Сборка и запуск (GPU)
```bash
docker-compose build helmet_gpu
docker-compose up helmet_gpu
```

### Сборка и запуск (CPU)
```bash
docker-compose build helmet_cpu
docker-compose up helmet_cpu
```

> Контейнеры монтируют `./violations:/app/violations` — кадры сохраняются и видны на хосте.

---

## Конфигурация (через `.env` / `config.py`)

- `VIDEO_SOURCE` — `0` / `"video.mp4"` / `http://...` / `rtsp://...`
- `MODEL_PATH` — путь к `best.pt`
- `PROCESS_FPS` — количество обрабатываемых кадров в секунду
- `VIOLATION_SECONDS` — длительность отсутствия каски для фиксации
- `COOLDOWN_SECONDS` — пауза между фиксациями для одного трека
- `OUTPUT_DIR` — папка для сохранённых кадров
- `CONF_PERSON`, `CONF_HEAD`, `CONF_HELMET` — confidence thresholds
- `DEVICE` — `cpu` или `cuda`

---

## Как работает

1. Захват кадра (OpenCV) из `VIDEO_SOURCE`.
2. Детекция YOLO — классы `person`, `head`, `helmet`.
3. IoU-трекер присваивает каждому человеку уникальный ID.
4. Логика нарушения: если есть `head`, но нет `helmet` дольше `VIOLATION_SECONDS` → сохраняем кадр.
5. MJPEG-стрим отдаётся браузеру на `/video`.

---


## Оптимизация

- Уменьшить `PROCESS_FPS` для CPU.

---
