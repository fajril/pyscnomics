from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import os
import builtins
from contextvars import ContextVar

root_print = builtins.print
user_id_ctx: ContextVar[str] = ContextVar("user_id_ctx", default="anonymous")

def override_print(logger):
    def custom_print(*args, **kwargs):
        msg = " ".join(str(arg) for arg in args)
        logger.info(msg)
    builtins.print = custom_print

class UserLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        user_id = request.headers.get("X-User-ID", "anonymous")
        user_id_ctx.set(user_id) # Set context variable

        logger = logging.getLogger(f"psc-user-{user_id}")
        logger.setLevel(logging.INFO)

        # Buat file handler per user
        log_path = f"logs/user_{user_id}.log"

        if not logger.handlers:
            handler = logging.FileHandler(log_path)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Inject ke request.state
        request.state.logger = logger

        #print new initialization message if file is new
        if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
            logger.info(f"Logger for user {user_id}")

        # Override print agar semua print(...) diarahkan ke logger
        override_print(request.state.logger)        

        response = await call_next(request)
        # logger.info(f"{request.method} {request.url} → {response.status_code}")
        return response
