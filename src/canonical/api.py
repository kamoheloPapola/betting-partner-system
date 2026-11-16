"""Read-only loopback dashboard/API over the canonical database."""
from pathlib import Path
import sqlite3
from fastapi import FastAPI,HTTPException,Query
from fastapi.responses import HTMLResponse,Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from .clock import CanonicalError
from .store import Store
from .service import health
from .cli import forecast_rows
from .provider import LEAGUES

ASSETS=Path(__file__).with_name("assets")


def create_app(database):
    database=Path(database)
    app=FastAPI(title="Betting Partner",docs_url=None,redoc_url=None)
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=["127.0.0.1","localhost"])

    @app.middleware("http")
    async def security_headers(request,call_next):
        response=await call_next(request)
        response.headers["Content-Security-Policy"]="default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"
        response.headers["X-Content-Type-Options"]="nosniff"
        response.headers["Cache-Control"]="no-store"
        return response

    def read(action):
        if not database.exists():
            return {"state":"NEEDS_SETUP","reasons":["DATABASE_NOT_INITIALIZED"],"forecasts":[]}
        try:
            with Store(database,readonly=True) as store:return action(store)
        except CanonicalError as exc:raise HTTPException(409,detail=exc.code) from None
        except (OSError,sqlite3.Error,ValueError,KeyError):
            raise HTTPException(503,detail="CANONICAL_DATA_UNAVAILABLE") from None

    @app.get("/",response_class=HTMLResponse)
    def index():
        return (ASSETS/"index.html").read_text(encoding="utf-8")

    @app.get("/app.js")
    def javascript():
        return Response((ASSETS/"app.js").read_text(encoding="utf-8"),media_type="application/javascript")

    @app.get("/style.css")
    def stylesheet():
        return Response((ASSETS/"style.css").read_text(encoding="utf-8"),media_type="text/css")

    @app.get("/api/health")
    def model_health():
        return read(health)

    @app.get("/api/forecasts")
    def forecasts(league:str|None=None,limit:int=Query(200,ge=1,le=1000)):
        if league is not None and league not in LEAGUES:raise HTTPException(422,detail="UNKNOWN_LEAGUE")
        return read(lambda store:{"forecasts":forecast_rows(store,league,limit)})

    @app.get("/api/forecasts/{fixture_id}")
    def forecast(fixture_id:int):
        def one(store):
            row=next((r for r in forecast_rows(store,limit=10000) if r["fixture_id"]==fixture_id),None)
            if row is None:raise HTTPException(404,detail="FROZEN_FORECAST_NOT_FOUND")
            return row
        return read(one)

    return app

