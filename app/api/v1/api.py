from fastapi import APIRouter
from app.api.v1.endpoints import login

api_router = APIRouter()
api_router.include_router(login.router, tags=["login"])
# api_router.include_router(users.router, prefix="/users", tags=["users"])
from app.api.v1.endpoints import scans
api_router.include_router(scans.router, prefix="/scans", tags=["scans"])
