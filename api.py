from flask import Flask, request, jsonify
from gradio_client import Client, handle_file

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get image file from request
        if 'image' not in request.files:
            return jsonify({"error": "Image file is required"}), 400

        image_file = request.files['image']

        # Save the image to a temporary location
        image_path = f"/tmp/{image_file.filename}"
        image_file.save(image_path)

        # Initialize the Gradio Client
        client = Client("auzalfred/Jewelry_Design_Gen")

        # Call the Gradio Space API
        result = client.predict(
            image=handle_file(image_path),
            api_name="/process_and_generate"
        )

        # Return the result
        return jsonify({
            "analysis": result[0],
            "generated_image": result[1]['url']  # Assuming the result is a URL
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)