"""Auth API routes — signup / login / refresh / logout / me."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .. import auth

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
_bearer = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=8)
    full_name: str = ""


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, description="Username or email")
    password: str = Field(..., min_length=1)


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    username: str
    email: str
    full_name: str
    role: str
    created_at: str
    last_login: str | None


class MessageResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/signup", response_model=TokenResponse, status_code=201)
def signup(body: SignupRequest) -> TokenResponse:
    pair = auth.signup(
        username=body.username,
        email=body.email,
        password=body.password,
        full_name=body.full_name,
    )
    return TokenResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        token_type=pair.token_type,
        expires_in=pair.expires_in,
    )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, request: Request) -> TokenResponse:
    ip = request.client.host if request.client else "0.0.0.0"
    pair = auth.login(body.username, body.password, ip=ip)
    return TokenResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        token_type=pair.token_type,
        expires_in=pair.expires_in,
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest) -> TokenResponse:
    pair = auth.refresh_tokens(body.refresh_token)
    return TokenResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        token_type=pair.token_type,
        expires_in=pair.expires_in,
    )


@router.post("/logout", response_model=MessageResponse)
def logout(
    user: auth.UserRecord = Depends(auth.get_current_user),
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> MessageResponse:
    if credentials:
        auth.logout(credentials.credentials)
    return MessageResponse(message="Logged out successfully.")


@router.get("/me", response_model=UserResponse)
def me(user: auth.UserRecord = Depends(auth.get_current_user)) -> UserResponse:
    return UserResponse(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        created_at=user.created_at,
        last_login=user.last_login,
    )
