import base64

# Paste the "image" value from the Postman response here
b64_string = "decoded_b64_value"

with open("output.png", "wb") as f:
    f.write(base64.b64decode(b64_string))

print("Image saved to output.png")