import gradio as gr
import torch
from threading import Thread
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TextStreamer, TextIteratorStreamer
from datetime import datetime
from daily_tickers import get_tickers

# Get the tickers at the start of the application
formatted_data = get_tickers()
print(formatted_data)
# Add a global variable for tracking the last update date
last_update_date = datetime.today().strftime("%Y-%m-%d")

# Fine-tuned Huggingface hosted model
model_name = "jackma-00/lora_model_1b"

# Model's parameters
max_seq_length = 2048
dtype = None
load_in_4bit = True
system_message = "You are a financial consultant. Answer your client's questions using yesterday's closing aggregates for the following key tickers: {}".format(
    formatted_data
)
max_tokens = 1024
temperature = 1.5
top_p = 0.95

# Load model and tokenizer from pretrained
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable native 2x faster inference
FastLanguageModel.for_inference(model)

# Create a text streamer
text_streamer = TextIteratorStreamer(
    tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
)


# Define inference function
def respond(message, history):
    global formatted_data, last_update_date, system_message

    # Check if the date has changed
    current_date = datetime.today().strftime("%Y-%m-%d")
    if current_date != last_update_date:
        # Update the tickers and the system message
        formatted_data = get_tickers()
        print(formatted_data)
        last_update_date = current_date
        system_message = "You are a financial consultant. Answer your client's questions using yesterday's closing aggregates for the following key tickers: {}".format(
            formatted_data
        )

    # Add system message
    messages = [{"role": "system", "content": system_message}]

    # Include chat history
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    # Lastly append user's message
    messages.append({"role": "user", "content": message})

    # Tokenize the input
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate arguments
    generate_kwargs = dict(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=max_tokens,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # Predict
    partial_message = ""
    for new_token in text_streamer:
        if new_token != "<":
            partial_message += new_token
            yield partial_message


# Define Gradio UI
gr = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(
        placeholder="Ask me financial guidance", container=False, scale=7
    ),
    title="Your financial consultant",
    description="Ask anything regarding finance, I will respond you based on the latest closing of major key tickers.",
    theme="soft",
    examples=[
        "Should I invest in gold today?",
        "How can I allocate additional 10k to my portfolio?",
        "What could be a good diversification strategy?",
    ],
)

if __name__ == "__main__":
    gr.launch(debug=True)
