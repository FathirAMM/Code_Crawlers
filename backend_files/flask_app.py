from flask import Flask, jsonify, render_template, request
from finetuning import fine_tune_model
from rag_application import answer_question, get_urls
from flask_cors import CORS
from chat_with_pdf import answer_pdf, answer_only
from inject_vector import inject_to_faiss, inject_to_pinecone
import tempfile
import os
from feed_back import save_feedback_to_csv, save_report_to_csv
from own_model import answer_question_own_model
from inject_csv_url import inject_url, inject_csv
from report_gen import report_generation
from evaluation import evaluate_rag_using_ragas

app = Flask(__name__)
CORS(app)


@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello, World!"})


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/answer_question", methods=["POST"])
def api_answer_question():
    data = request.json
    print(data)
    question = data["question"]
    model = data["model"]
    database_name = data["database"]
    api_key = data["apikey"]
    options = ["gpt-4", "gpt-3.5-turbo", "mistral", "llama2-7b", "gemma-7b", "claude-2"]
    if model in options:
        try:
            result, relevant_images = answer_question(question, model, database_name)

            try:
                urls = get_urls(question)
            except Exception as e:
                print({"error": str(e)})

            return jsonify(
                {"result": result, "relevant_images": relevant_images, "urls": urls}
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    else:
        try:
            result, relevant_images = answer_question_own_model(
                question, model, database_name, api_key
            )
            urls = get_urls(question)
            return jsonify(
                {"result": result, "relevant_images": relevant_images, "urls": urls}
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400


@app.route("/api/finetune", methods=["POST"])
def finetune():
    print(request.files["file"])
    print(request.form["suffixName"])
    print(request.form["apiKey"])
    file = request.files["file"]
    suffix = request.form["suffixName"]
    api_key = request.form["apiKey"]
    try:
        model_id = fine_tune_model(file, suffix, api_key)
        return jsonify({"model_id": model_id})
    except Exception as e:
        print({"error": str(e)})
        return jsonify({"error": str(e)}), 500


@app.route("/api/inject_to_vector_db", methods=["POST"])
def inject_to_vector_db():

    print(request.form["database"])
    file = request.files["file"]

    database_choice = request.form["database"]
    url = request.form["url"]
    print(url)

    if file and database_choice:
        print("file sent")
        temp_dir = tempfile.TemporaryDirectory()
        temp_pdf_path = os.path.join(temp_dir.name, file.filename)
        file.save(temp_pdf_path)
        print(temp_pdf_path)
        file_extension = temp_pdf_path.split(".")[-1].lower()
        print(file_extension)
        if file_extension == "pdf":
            try:
                if database_choice == "faiss":
                    print("faiss")
                    inject_to_faiss(temp_pdf_path)
                else:
                    print("pinecone")
                    inject_to_pinecone(temp_pdf_path)
                return jsonify(
                    {
                        "message": f"Data has been successfully injected into {database_choice}."
                    }
                )
            except Exception as e:
                print({"error": str(e)})
                return jsonify({"error": str(e)}), 500
        else:
            try:
                inject_csv(temp_pdf_path, database_choice)
                return jsonify(
                    {
                        "message": f"Data has been successfully injected into {database_choice}."
                    }
                )
            except Exception as e:
                print({"error": str(e)})
                return jsonify({"error": str(e)}), 500

    elif url and database_choice:
        try:
            print(url)
            inject_url(url, database_choice)
            return jsonify(
                {
                    "message": f"Data has been successfully injected into {database_choice}."
                }
            )
        except Exception as e:
            print({"error": str(e)})
            return jsonify({"error": str(e)}), 500


@app.route("/api/chat_with_pdfs", methods=["POST"])
def chat_with_pdfs():
    question = request.form["question"]
    print(question)
    file = request.files["file"]
    print(file)
    try:
        if file and question:
            # Save the file temporarily
            temp_dir = tempfile.TemporaryDirectory()
            temp_pdf_path = os.path.join(temp_dir.name, file.filename)
            file.save(temp_pdf_path)
            print(temp_dir.name)
            print(temp_pdf_path)
            print(question)
            answer, images_base64 = answer_pdf(question, temp_dir.name)
            print(answer)

            temp_dir.cleanup()
            return jsonify({"result": answer, "relevant_images": images_base64})
        else:
            print("no file")
            answer, images_base64 = answer_only(question)
            return jsonify({"result": answer, "relevant_images": images_base64})
    except Exception as e:
        print({"error": str(e)})
        return jsonify({"error": str(e)}), 500


@app.route("/api/submit_feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json()
    print(data)
    if "message_id" not in data or "response" not in data or "feedback" not in data:
        return jsonify({"error": "Invalid data format"}), 400

    message_id = data["message_id"]
    response = data["response"]
    feedback = data["feedback"]

    save_feedback_to_csv(message_id, response, feedback, "feedback_data.csv")

    return jsonify({"success": True})


@app.route("/api/submit_report", methods=["POST"])
def submit_report():
    data = request.get_json()
    print(data)
    if "message_id" not in data or "response" not in data:
        return jsonify({"error": "Invalid data format"}), 400

    message_id = data["message_id"]
    response = data["response"]

    save_report_to_csv(message_id, response)

    return jsonify({"success": True})


@app.route("/api/report_generation", methods=["POST"])
def report_generate():
    data = request.get_json()
    number = data["number"]
    # number = request.form["number"]
    print(number)
    no_of_msg = int(number)
    print(no_of_msg)
    if no_of_msg > 5:
        no_of_msg == 5
    else:
        no_of_msg = no_of_msg
    try:
        if no_of_msg:
            if report_generation(no_of_msg):
                return jsonify({"message": f"Report was successfully generated"})
    except Exception as e:
        print({"error": str(e)})
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluation", methods=["POST"])
def evaluate_model():
    model = request.form["model_name"]
    api_key = request.form["api_key"]
    file_path = ""
    if model and api_key:
        try:
            results = evaluate_rag_using_ragas(model, api_key, file_path)
            return jsonify({"result": results})

        except Exception as e:
            print({"error": str(e)})
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
