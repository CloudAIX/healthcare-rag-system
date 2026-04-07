"""API security — API key auth, JWT tokens, and rate limiting."""
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

# --- Configuration ---

SECRET_KEY = os.getenv("RAG_SECRET_KEY", secrets.token_urlsafe(32))
API_KEY = os.getenv("RAG_API_KEY", "hrag-dev-key-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("RAG_JWT_EXPIRE_MINUTES", "60"))

ALLOWED_ORIGINS = os.getenv(
    "RAG_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8504"
).split(",")

# --- Rate Limiter ---

limiter = Limiter(key_func=get_remote_address)

# --- Schemas ---

class TokenRequest(BaseModel):
    api_key: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

# --- API Key Auth ---

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)) -> str:
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "API key"})
    return key

# --- JWT Auth ---

bearer_scheme = HTTPBearer(auto_error=False)

def create_access_token(subject: str = "api-client") -> tuple[str, int]:
    expires = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": subject, "exp": expires, "iat": datetime.now(timezone.utc)}
    token = jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token, JWT_EXPIRE_MINUTES * 60

async def verify_jwt(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> dict:
    if not credentials:
        raise HTTPException(401, "Missing authorization", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})

# --- Combined Auth (API key OR JWT) ---

async def authenticate(
    api_key: str = Security(api_key_header),
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> str:
    """Accept either a valid API key or a valid JWT bearer token."""
    # Try API key first
    if api_key and secrets.compare_digest(api_key, API_KEY):
        return "api-key"
    # Try JWT
    if credentials:
        try:
            payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return f"jwt:{payload.get('sub', 'unknown')}"
        except JWTError:
            pass
    raise HTTPException(
        401,
        "Valid API key (X-API-Key header) or Bearer token required",
        headers={"WWW-Authenticate": "Bearer"},
    )
