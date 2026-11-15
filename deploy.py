from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from hf_sentiment import SentimentPrediction, load_sentiment_model

WEB_APP_PATH = Path(__file__).resolve().parent / "static" / "index.html"


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


class SentimentModel(Protocol):
    def predict(self, text: str) -> SentimentPrediction: ...


def create_app(
    model_loader: Callable[[], SentimentModel] = load_sentiment_model,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model_loader()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(WEB_APP_PATH)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(inp: TextIn, request: Request) -> SentimentPrediction:
        text = inp.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="text must not be blank")
        return request.app.state.model.predict(text)

    return app


app = create_app()
