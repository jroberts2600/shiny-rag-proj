import shiny.experimental as exp
from shiny import App, ui, render, reactive
import os
from utils import process_documents, query_model, get_all_models, verify_api_key_sync, MAX_FILE_SIZE, is_ollama_available
from io import BytesIO

# Ensure the 'documents' directory exists
if not os.path.exists("documents"):
    os.makedirs("documents")

def log_active_model(model_info):
    print(f"Active model: {model_info}")

app_ui = ui.page_fluid(
    ui.include_css("www/styles.css"),
    ui.tags.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap"),
    ui.div(
        ui.h2("RAG Search Pipeline"),
        class_="header"
    ),
    ui.div(
        ui.div(
            ui.h3("Model Selection"),
            ui.input_select("model_category", "Select Model Category", choices=list(get_all_models().keys())),
            ui.output_ui("model_selection"),
            ui.output_ui("api_key_input"),
            ui.div(ui.output_text("api_key_status"), class_="status-message"),
            ui.div(ui.output_text("model_status"), class_="status-message"),
            class_="content-card"
        ),
        ui.div(
            ui.h3("Document Upload"),
            ui.div(
                ui.input_file("uploaded_files", "Upload documents", multiple=True, 
                              accept=[".docx", ".pdf", ".html", ".htm", ".txt", ".png", ".jpeg", ".jpg", ".pptx", ".csv"]),
                class_="file-input"
            ),
            ui.div(ui.output_text("status"), class_="status-message"),
            class_="content-card"
        ),
        ui.div(
            ui.h3("Processed Documents"),
            ui.output_text("processed_docs"),
            class_="content-card"
        ),
        ui.div(
            ui.h3("Ask a Question"),
            ui.input_text("query", "Enter your question about the documents"),
            ui.div(
                ui.div(ui.output_text("response"), id="response"),
                ui.input_action_button("copy_button", "Copy", class_="copy-btn", onclick="copyToClipboard()"),
                class_="response-container"
            ),
            class_="content-card"
        ),
        ui.input_action_button("clear_vector_store", "Clear Vector Store", class_="btn btn-secondary"),
    ),
    ui.output_ui("loading_indicator"),
    ui.tags.script("""
    function copyToClipboard() {
        var responseText = document.getElementById('response').innerText;
        navigator.clipboard.writeText(responseText).then(function() {
            var copyBtn = document.getElementById('copy_button');
            copyBtn.innerText = 'Copied!';
            setTimeout(function() {
                copyBtn.innerText = 'Copy';
            }, 2000);
        }, function(err) {
            console.error('Could not copy text: ', err);
        });
    }
    """)
)

def server(input, output, session):
    db = reactive.Value(None)
    status_message = reactive.Value("")
    api_key_status_message = reactive.Value("")
    model_status_message = reactive.Value("")
    processed_documents = reactive.Value([])
    is_loading = reactive.Value(False)

    @output
    @render.ui
    def model_selection():
        category = input.model_category()
        models = get_all_models()[category]
        choices = ["Choose Model"] + models
        return ui.input_select("selected_model", "Select Model", choices=choices)

    @output
    @render.ui
    def api_key_input():
        category = input.model_category()
        if category in ["OpenAI", "Claude", "HuggingFace"]:
            env_var_name = "OPENAI_API_KEY" if category == "OpenAI" else "ANTHROPIC_API_KEY" if category == "Claude" else "HUGGINGFACEHUB_API_TOKEN"
            env_api_key = os.environ.get(env_var_name)
            
            if env_api_key:
                return ui.div(
                    ui.p(f"{category} API key detected from environment variable."),
                    ui.input_text("api_key", "API Key", value=env_api_key),
                    ui.tags.script("""
                        document.getElementById('api_key').type = 'password';
                    """)
                )
            else:
                return ui.div(
                    ui.input_text("api_key", f"Enter {category} API Key"),
                    ui.tags.script("""
                        document.getElementById('api_key').type = 'password';
                    """),
                    ui.p(f"You can also set {env_var_name} environment variable.")
                )
        return ui.div()

    @reactive.Effect
    @reactive.event(input.model_category, input.selected_model, input.api_key)
    def update_model_status():
        category = input.model_category()
        model = input.selected_model()
        api_key = input.api_key() if hasattr(input, 'api_key') else None

        if model == "Choose Model":
            api_key_status_message.set("")
            model_status_message.set("Please select a model")
            return

        is_loading.set(True)
        try:
            if category in ["OpenAI", "Claude", "HuggingFace"]:
                status, _, detected_api_type = verify_api_key_sync(api_key)
                api_key_status_message.set(status)
            elif category == "Ollama":
                if not is_ollama_available():
                    raise ValueError("Ollama is not available. Please ensure it's running and accessible.")
                api_key_status_message.set("No API key required for Ollama.")
            else:
                api_key_status_message.set("No API key required for this model.")
            
            model_status_message.set(f"Active model: {model} ({category})")
            log_active_model(f"{model} ({category})")
        except Exception as e:
            api_key_status_message.set(f"Error: {str(e)}")
            model_status_message.set("Active model: Unknown")
            log_active_model("Unknown (Error)")
        finally:
            is_loading.set(False)

    @reactive.Effect
    @reactive.event(input.uploaded_files)
    def process_files():
        status_message.set("Processing documents...")
        if not input.uploaded_files():
            status_message.set("Please upload documents.")
            return None

        api_key = input.api_key() if hasattr(input, 'api_key') else None
        uploaded_files = input.uploaded_files()

        files = []
        for file_info in uploaded_files:
            file_name = file_info["name"]
            file_path = file_info["datapath"]
            file_size = os.path.getsize(file_path)
            if file_size > MAX_FILE_SIZE:
                status_message.set(f"Error: {file_name} exceeds the maximum size limit of {MAX_FILE_SIZE / (1024 * 1024)} MB")
                return None
            with open(file_path, "rb") as f:
                file_content = f.read()
                file_like_object = BytesIO(file_content)
                file_like_object.name = file_name
                files.append(file_like_object)

        is_loading.set(True)
        try:
            status_message.set("Processing documents... Please wait.")
            db_result, processed_files = process_documents(files, api_key)
            db.set(db_result)
            processed_documents.set(processed_files)
            status_message.set("Documents processed successfully! You can now ask a question.")
        except Exception as e:
            status_message.set(f"Error processing documents: {str(e)}")
        finally:
            is_loading.set(False)

    @reactive.Effect
    @reactive.event(input.clear_vector_store)
    def clear_vector_store():
        try:
            if db.get():
                db.get().delete_collection()
                db.set(None)
                processed_documents.set([])
                status_message.set("Vector store cleared successfully!")
            else:
                status_message.set("No vector store to clear.")
        except Exception as e:
            status_message.set(f"Error clearing vector store: {str(e)}")

    @reactive.Calc
    def generate_response():
        query = input.query()
        if query and db.get():
            status_message.set(f"Generating response...")
            is_loading.set(True)
            try:
                api_key = input.api_key() if hasattr(input, 'api_key') else None
                model = input.selected_model()
                category = input.model_category()
                
                if model == "Choose Model":
                    raise ValueError("Please select a model before querying.")
                
                if category in ["OpenAI", "Claude", "HuggingFace"] and not api_key:
                    raise ValueError(f"API key is required for {category} models.")
                
                response = query_model(db.get(), query, api_key, model)
                status_message.set("Response generated successfully.")
                return response
            except ValueError as e:
                error_message = str(e)
                status_message.set(f"Error: {error_message}")
                return f"Error: {error_message}"
            except Exception as e:
                error_message = str(e)
                status_message.set(f"Unexpected error: {error_message}")
                return f"Unexpected error: {error_message}"
            finally:
                is_loading.set(False)
        return ""

    @output
    @render.text
    def api_key_status():
        return api_key_status_message.get()

    @output
    @render.text
    def status():
        return status_message.get()

    @output
    @render.text
    def response():
        return generate_response()

    @output
    @render.text
    def processed_docs():
        docs = processed_documents.get()
        return "\n".join(docs) if docs else "No documents processed yet."

    @output
    @render.text
    def model_status():
        return model_status_message.get()

    @output
    @render.ui
    def loading_indicator():
        if is_loading.get():
            return ui.div(
                ui.tags.div(class_="loading-spinner"),
                "Loading...",
                class_="loading-overlay"
            )
        return ui.div()

app = App(app_ui, server)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)