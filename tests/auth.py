"""Test-only token validator: the bearer token is the user's username."""

from django.contrib.auth import get_user_model


def authenticate_token(token: str):
    User = get_user_model()
    user = User.objects.filter(username=token).first()
    if user is None:
        raise ValueError("Invalid token")
    return user
