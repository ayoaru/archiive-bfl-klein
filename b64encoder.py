import base64

with open("cc_zip_washed_grey.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")
    print(b64)