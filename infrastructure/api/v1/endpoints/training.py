"""Endpoint to trigger training."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/train")
def train(payload: dict):
    # TODO: wire training orchestration
    return {"status": "queued"}
