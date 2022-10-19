import uvicorn

class App:
    ...

app = App()

if __name__ == "__main__":
    uvicorn.run("backend_sim:app", host="127.0.0.1", port=80, log_level="info")
