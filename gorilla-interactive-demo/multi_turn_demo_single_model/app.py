import gradio as gr
from backend import get_handler
import time
import uuid
import json
import os
import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://r112358:Ef8C6ILA5LT7GE06@cluster0.qsbfw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
cluster = MongoClient(uri, server_api=ServerApi('1'))
try:
    cluster.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
db = cluster["test"]
collection = db["test"]

custom_css = """
/* Highlight the entire message box for the bot */
.bot-highlight {
    background-color: yellow !important;
    padding: 0px;
    border-radius: 8px;
}
"""

# Initialize test_entry and handler
def initialize_empty_test_entry():
    return {
        "initial_config": {},
        "involved_classes": [],
        "id": str(uuid.uuid4()),  # Generate unique ID
        "question": [],
        "function": []
    }

test_entry = initialize_empty_test_entry()

inference_data = {"message": []}  # Do not change this

def open_json(filename):
    with open(filename, 'r') as file:
        data = []
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line}")
                print(f"Error message: {e}")
    return data

def format_questions(question):
    questions = """<div style="text-align: left; width: 500px;">\n"""
    for i, q in enumerate(question):
        questions += f"<p style='margin: 10px 0;'><b>Turn {i + 1} Question:</b> <br>\n {q[0]['content']}<br>\n"
    return questions

def format_solutions(solution):
    solutions = """<div style="text-align: left;">"""
    for i, s in enumerate(solution):
        solutions += f"<b>Turn {i+1} Response</b> \n\n ```python\n"
        for c in s:
            solutions += c + "\n"
        solutions += "\n```\n\n"
    return solutions

def process(df, questions, solutions):
    df['involved_classes'] = [questions[int(df['id'][i].split("_")[-1])]['involved_classes'] for i in df.index]
    df['question'] = [questions[int(df['id'][i].split("_")[-1])]['question'] for i in df.index]
    df['ground_truth'] = [solutions[int(df['id'][i].split("_")[-1])]['ground_truth'] for i in df.index]
    df['question'] = df['question'].apply(format_questions)
    df['ground_truth'] = df['ground_truth'].apply(format_solutions)
    return df

api_info = pd.read_csv("api_info.csv")
questions = open_json('BFCL_v3_multi_turn_base.json')
solutions = open_json('BFCL_v3_multi_turn_base_sol.json')
api_samples = pd.read_csv('samples_qa.csv')
api_samples = process(api_samples, questions, solutions)

# Define available models and categories
models = ["gpt-4o-mini-2024-07-18-FC", "gpt-4o-2024-08-06-FC", "gpt-4o-mini-2024-07-18-FC", "gpt-4-turbo-2024-04-09-FC", "gpt-3.5-turbo-0125-FC"]
categories = ["GorillaFileSystem", "MathAPI", "MessageAPI", "TwitterAPI", "TicketAPI", "TradingBot", "TravelAPI", "VehicleControlAPI"]
current_model = models[0]
current_category = categories[0]

initial_chat_history = [
    {"role": "user", "content": "Hi, can you help me with some tasks?"},
    {"role": "assistant", "content": "Hello there! How can I assist you today?"},
]

def get_initial_state():
    return initial_chat_history

DEFAULT_TEMPERATURE = 0.7

# Initialize handler
handler = get_handler(models[0], DEFAULT_TEMPERATURE)

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
        test_entry["question"] = [{"role": "user", "content": message["text"]}]
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def equalize_and_zip(list1, list2):
    # Determine the maximum length of the two lists
    if list1 == None:
        list1 = []
    if list2 == None:
        list2 = []
        
    if isinstance(list1, str):
        list1 = [list1]
    if isinstance(list2, str):
        list2 = [list2]
    
    max_len = max(len(list1), len(list2))
    
    # Extend both lists to the same length by appending None
    list1.extend([None] * (max_len - len(list1)))
    list2.extend([None] * (max_len - len(list2)))
    
    # Zip the lists together
    return list(zip(list1, list2))


def bot(history: list):
    global inference_data, test_entry
    
    for item in handler.inference(test_entry, inference_data):
        # print("item: ", item)
        if item[0] == "regular":
            responses_results = equalize_and_zip(item[1], item[2])
            for (model_res, exec_res) in responses_results:
                if model_res is not None:
                    response = model_res
                    history.append({"role": "assistant", "content": "Model Response: "})
                    for character in response:
                        history[-1]["content"] += character
                        time.sleep(0.01)
                        yield history
                if exec_res is not None:
                    response = exec_res
                    history.append({"role": "assistant", "content": "<span class='bot-highlight'> Model Execution: </span>"})
                    for character in response:
                        history[-1]["content"] = history[-1]["content"][0:-7] + character + "</span>"
                        time.sleep(0.01)
                        yield history
        elif item[0] == 'summary':
            response = item[1]
            if response is not None:
                history.append({"role": "assistant", "content": "Summary: "})
                for character in response:
                    history[-1]["content"] += character
                    time.sleep(0.01)
                    yield history
        elif item[0] == "final":
            inference_data = item[2]
        time.sleep(0.05)

# Function to assign new unique ID on restart or at the start
def restart_chat(history):
    if len(history) > 2:  
        document = {"model": current_model,
                    "category": current_category,
                    "temperature": temperature_slider.value,
                    "history": history}
        collection.insert_one(document)
    global test_entry
    test_entry["id"] = str(uuid.uuid4())  # Reinitialize test_entry with new ID
    #update_handler(model_dropdown.value, temperature_slider.value)
    print("test: entry", test_entry)

    return [{"role": "user", "content": "Hi, can you help me with some tasks?"}, 
            {"role": "assistant", "content": "Hello there! How can I assist you today?"}]

# Function to report an issue
def report_issue():
    return gr.Info("Thank you for reporting the issue. Our team will look into it.")

# Update handler when model or temperature is changed
def update_handler(model, temp_slider, history):
    print("update handler: ", model, temp_slider)
    global handler
    global current_model 
    handler = get_handler(model, temp_slider)  # Reinitialize handler with new model and temperature
    restart_history = restart_chat(history)
    current_model = model
    return model, restart_history

# Update involved_classes and load config based on category
def updatecurrent_category_and_load_config(category):
    global test_entry
    
    print("update category: ", category)
    global current_category 
    current_category = category

    # Update involved_classes
    test_entry["initial_config"] = {category: {}}
    test_entry["involved_classes"] = [category]

    # Load the JSON file from the config folder corresponding to the category
    config_path = os.path.join("config", f"{category}.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as config_file:
            data = json.load(config_file)
            test_entry["function"] = data.copy()  # Load JSON content into test_entry["function"]
            # print("test entry function: ", test_entry["function"])
    return category

def load_example(example):
    if example == "Example 1 - GFSFileSystem":
        return models[0], 0.8, categories[0], "Move final_report.pdf' within document directory to 'temp' directory in document. Make sure to create the directory"
    elif example == "Example 2 - TradingBot":
        return models[1], 0.9, categories[5], "I'm contemplating enhancing my investment portfolio with some tech industry assets, and I've got my eye on Nvidia Corp. I'm keen to know its current stock price, and would appreciate if you could source this information for me."
    elif example == "Example 3 - TravelAPI":
        return models[2], 0.7, categories[6], "As I plan a getaway, I'm curious about all the airports available for my travel. Would you share that information with me?"
    return models[0], 0.7, categories[0], "Move final_report.pdf' within document directory to 'temp' directory in document. Make sure to create the directory"

# Add logic to load examples when example buttons are clicked
def load_example_and_update(example):
    # Load the example configuration
    model, temp, category, message = load_example(example)
    updatecurrent_category_and_load_config(category)
    # Update the interface components
    return model, temp, category, message

# initialize test entry with default configurations
updatecurrent_category_and_load_config(categories[0])

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Multiturn LLM Chat Interface")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Configuration")

            model_dropdown = gr.Dropdown(choices=models, label="Select Model", value=models[0], interactive=True)
            temperature_slider = gr.Slider(0, 1, value=DEFAULT_TEMPERATURE, label="Temperature", interactive=True)
            category_dropdown = gr.Dropdown(choices=categories, label="Select Category", value=categories[0], interactive=True)
        
        with gr.Column(scale=2):
                
            # chatbot = gr.Chatbot(value=initial_chat_history, elem_id="chatbot", bubble_full_width=False, type="messages")
            chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages")

            with gr.Row():
                restart_btn = gr.Button("Restart")
                report_btn = gr.Button("Report Issue")
                
            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_count="multiple",
                placeholder="Enter message or upload file...",
                show_label=False,
            )

            demo.load(lambda: get_initial_state(), outputs=chatbot)
            chat_msg = chat_input.submit(
                add_message, [chatbot, chat_input], [chatbot, chat_input]
            )
            bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
            bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

            chatbot.like(print_like_dislike, None, None, like_user_message=True)
            
            with gr.Row():
                example_btn1 = gr.Button("Example 1 - GFSFileSystem")
                example_btn2 = gr.Button("Example 2 - TradingBot")
                example_btn3 = gr.Button("Example 3 - TravelAPI")

    gr.Markdown("<br><br>")

    for category in categories:
        with gr.Tab(category):
            with gr.Group():
                category_info = api_info[api_info["Class Name"] == category]
                with gr.Accordion("Function description", open=False):
                    with gr.Group():
                        for i in range(len(category_info)):
                            with gr.Accordion(category_info.iloc[i]["Function Name"], open=False):
                                gr.Markdown(category_info.iloc[i]["Description"])

            # Sample demo, limit 5 per categories
            samples = [[sample['question'], sample['ground_truth']] for _, sample in api_samples.iterrows() if category in sample['involved_classes']][:5]
            gr.Dataset(
                    components=[gr.HTML(), gr.Markdown()],
                    headers= ["Prompt", "API Use"],
                    samples= samples
                )

    # Update handler when the model or temperature is changed
    model_dropdown.change(
        update_handler, 
        [model_dropdown, temperature_slider, chatbot], 
        [model_dropdown, chatbot]
    )

    # Update category and load config when a category is selected
    category_dropdown.change(
        updatecurrent_category_and_load_config,
        inputs=category_dropdown,
        outputs=category_dropdown
    )

    # Set up the event handler for the restart button to reset the chat and test_entry
    restart_btn.click(restart_chat, [chatbot], [chatbot])
    report_btn.click(report_issue, None, None)
    
    example_btn1.click(load_example_and_update, inputs=example_btn1, 
                       outputs=[model_dropdown, temperature_slider, category_dropdown, chat_input])
    example_btn2.click(load_example_and_update, inputs=example_btn2, 
                       outputs=[model_dropdown, temperature_slider, category_dropdown, chat_input])
    example_btn3.click(load_example_and_update, inputs=example_btn3, 
                       outputs=[model_dropdown, temperature_slider, category_dropdown, chat_input])

print("temperature_slider: ", temperature_slider.value)
demo.launch(share=True)




