print("Starting test app...")

from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World!"

if __name__ == "__main__":
    print("Server starting on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=True)