"""serve_pic.py - a PNG file as a data URL for the chat API (the test scripts)."""
import base64


def picture_url(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
