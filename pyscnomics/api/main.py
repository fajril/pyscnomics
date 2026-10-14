"""
This file is utilized for routing the API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pyscnomics.api.errors import install_error_handlers
from pyscnomics.api.router import router


description = (
    """
    ## Description
    We welcome you to our library, PySCnomics 🚀.
    
    This package contains tailored functionalities for assessing economic feasibility of 
    oil and gas projects following the state-of-the-art Production Sharing Contract (PSC) 
    schemes in Indonesia.
    
    PySCnomics is the product of join research between Indonesia's Special Task Force for 
    Upstream Oil and Gas Business Activities (SKK Migas) and the Department of Petroleum 
    Engineering, Institut Teknologi Bandung (ITB).
    """
)


print('\t')
print('description = \n', description)

app = FastAPI(title="PySCnomics", description=description, version="1.4.0")
# Install handlers before CORS so the unexpected-failure middleware sits inside
# the CORS middleware and its 500 responses keep the allowed-origin headers.
install_error_handlers(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

app.include_router(router)
