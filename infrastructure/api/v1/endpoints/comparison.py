"""Endpoint for comparing two handwriting samples."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/compare")
def compare(payload: dict):
    # TODO: wire use case
    return {"result": "not_implemented"}
