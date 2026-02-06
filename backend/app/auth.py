from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UserContext:
    username: str
    token: str


def authenticate(username: str, password: str) -> UserContext | None:
    if username and password == "demo":
        return UserContext(username=username, token=f"token-{username}")
    return None
