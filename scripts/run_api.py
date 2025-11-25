"""Script to run the API (FastAPI + Uvicorn)."""

def main():
    try:
        from infrastructure.api.v1.main import create_app
        import uvicorn

        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print("Make sure dependencies are installed", e)


if __name__ == "__main__":
    main()
