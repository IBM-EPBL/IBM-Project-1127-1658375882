from fastapi import FastAPI
app = FastAPI(
    title="test",
    description="test",
    version="0.0.1",
)
if __name__ == "__main__":
    import uvicorn

uvicorn.run(
    "main:app",
    host="0.0.0.0",
    reload=True,
    port=3001,
)