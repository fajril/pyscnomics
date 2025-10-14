import os
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from router import webRouter
from psc_logger import UserLoggerMiddleware
import uvicorn

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # await create_db_and_tables()
#     yield


app = FastAPI(
#docs_url=None,      # Matikan Swagger UI di /docs
#redoc_url=None,     # Matikan ReDoc di /redoc
#openapi_url=None    # Opsional: matikan akses ke schema JSON di /openapi.json    
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    # allow_credentials=True,
)

#app.add_middleware(UserLoggerMiddleware)

root_home = Path(os.path.dirname(__file__), "static")

app.include_router(webRouter)


app.mount(
    "/app/assets",
    StaticFiles(directory=Path(root_home, "assets")),
    name="static",
)
templates = Jinja2Templates(directory=root_home)

@app.middleware("http")
async def vue_fallback(request: Request, call_next):
    response = await call_next(request)
    if response.status_code == 404:
        original_path = request.url.path
        return RedirectResponse(url=f"/app{original_path}")
    return response

@app.get("/schemas/psc-schema/{name}")
def get_user_schema(name: str):
    return FileResponse(f"./psc-schema/{name}.json")

@app.get("/app/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(path=Path(root_home, "favicon.ico"))

@app.get("/app/loader.css", include_in_schema=False)
async def loader():
    return FileResponse(path=Path(root_home, "loader.css"))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return RedirectResponse(url="/app", status_code=status.HTTP_302_FOUND)

@app.get("/app", response_class=HTMLResponse)
async def read_root2(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "message": "Hello, World!"}
    )

@app.get("/app/{full_path:path}", response_class=HTMLResponse)
async def read_root3(request: Request, full_path: str):
    if full_path == "loader.css":
        return FileResponse(path=Path(root_home, "loader.css"))
    elif full_path == "favicon.ico":
        return FileResponse(path=Path(root_home, "favicon.ico"))
    else:
        return templates.TemplateResponse(
            "index.html", {"request": request, "message": "Hello, World!"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000,                
                ssl_keyfile="key.pem",
               ssl_certfile="cert.pem")