from pathlib import Path
import cv2
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from contextlib import asynccontextmanager

from fastapi.staticfiles import StaticFiles

from app.processor import VideoProcessor
from app.config import Settings

settings = Settings()

logger = logging.getLogger("helmet-check")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

processor: VideoProcessor | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor
    logger.info("Starting VideoProcessor...")
    processor = VideoProcessor(settings.VIDEO_SOURCE, logger=logger)
    try:
        yield
    finally:
        logger.info("Stopping VideoProcessor...")
        if processor:
            processor.release()

app = FastAPI(lifespan=lifespan, title="Helmet Check")

@app.get("/", response_class=HTMLResponse)
def index():
    html = """
    <html><body>
      <h2>Helmet Check — demo</h2>
      <img src="/video" />
    </body></html>
    """
    return HTMLResponse(html)

def mjpeg_stream():
    while True:
        if processor is None:
            break
        frame = processor.read()
        if frame is None:
            continue
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpg.tobytes()
            + b"\r\n"
        )

@app.get("/video")
def video():
    return StreamingResponse(
        mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/health")
def health():
    return {"status": "ok"}

app.mount("/violations/static", StaticFiles(directory=settings.OUTPUT_DIR), name="violations_static")

@app.get("/violations/view", response_class=HTMLResponse)
def view_violations():
    html = "<h1>Violations</h1>"
    for p in sorted(Path(settings.OUTPUT_DIR).glob("*.jpg")):
        url = f"/violations/static/{p.name}"
        html += f'<div style="margin:10px;"><img src="{url}" style="max-width:300px;"><p>{p.name}</p></div>'
    return html
