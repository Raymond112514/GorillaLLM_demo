import gradio as gr
from backend import get_handler
import time
import uuid
import json
import os
import queue
import threading
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

test_entry_1 = initialize_empty_test_entry()
test_entry_2 = initialize_empty_test_entry()

inference_data_1 = {"message": []}  # Do not change this
inference_data_2 = {"message": []}  # Do not change this

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

# Load api info and samples
# sample_qa.csv contains a list of questions from base_multiturn ex: [multi_turn_base_41, ...]
api_info = pd.read_csv("api_info.csv")
questions = open_json('BFCL_v3_multi_turn_base.json')
solutions = open_json('BFCL_v3_multi_turn_base_sol.json')
api_samples = pd.read_csv('samples_qa.csv')
api_samples = process(api_samples, questions, solutions)

# Define available models and categories
models = ["gpt-4o-mini-2024-07-18-FC", "gpt-4o-2024-08-06-FC", "gpt-4o-mini-2024-07-18-FC", "gpt-4-turbo-2024-04-09-FC", "gpt-3.5-turbo-0125-FC"]
categories = ["GorillaFileSystem", "MathAPI", "MessageAPI", "TwitterAPI", "TicketAPI", "TradingBot", "TravelAPI", "VehicleControlAPI"]
current_model_1 = models[0]
current_category_1 = categories[0]
current_model_2 = models[0]
current_category_2 = categories[0]

initial_chat_history = [
    {"role": "user", "content": "Hi, can you help me with some tasks?"},
    {"role": "assistant", "content": "Hello there! How can I assist you today?"},
]

shared_queue = queue.Queue()

def get_initial_state():
    return initial_chat_history

DEFAULT_MODEL_1 = models[0]
DEFAULT_MODEL_2 = models[1]
DEFAULT_TEMPERATURE_1 = 0.7
DEFAULT_TEMPERATURE_2 = 0.4

# Initialize handler
handler_1 = get_handler(DEFAULT_MODEL_1, DEFAULT_TEMPERATURE_1)
handler_2 = get_handler(DEFAULT_MODEL_2, DEFAULT_TEMPERATURE_2)

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(history1, history2, message, target):
    
    if target in ["Model 1", "Both"]:
        for x in message["files"]:
            history1.append({"role": "user", "content": {"path": x}})
        if message["text"] is not None:
            history1.append({"role": "user", "content": message["text"]})
            test_entry_1["question"] = [{"role": "user", "content": message["text"]}]
            
    if target in ["Model 2", "Both"]:
        for x in message["files"]:
            history2.append({"role": "user", "content": {"path": x}})
        if message["text"] is not None:
            history2.append({"role": "user", "content": message["text"]})
            test_entry_2["question"] = [{"role": "user", "content": message["text"]}]
            
    return history1, history2, gr.MultimodalTextbox(value=None, interactive=False)

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

def consume_data(shared_queue):
    none_list = []
    while True:
        data = shared_queue.get()
        if data is None:
            if data in none_list:
                print("[Consumer] No more data to consume. Exiting.")
                break
            else:
                none_list.append(data)
        yield data

def bot(history1: list, history2: list, target):
    
    if target == "Model 1":
        gen = bot_response(history1, handler_1, test_entry_1, inference_data_1, "Model 1")
        print("gen: ", gen)
        while True:
            stop = True
            try:
                gen_his_1 = next(gen)
                stop = False
                yield gen_his_1, history2
            except StopIteration:
                pass
            if stop:
                break
    elif target == "Model 2":
        gen = bot_response(history2, handler_2, test_entry_2, inference_data_2, "Model 2")
        while True:
            stop = True
            try:
                gen_his_2 = next(gen)
                stop = False
                yield history1, gen_his_2
            except StopIteration:
                pass
            if stop:
                break
    elif target == "Both":
        gen1 = bot_response(history1, handler_1, test_entry_1, inference_data_1, "Model 1")
        gen2 = bot_response(history2, handler_2, test_entry_2, inference_data_2, "Model 2")
        while True:
            stop = True
            try:
                gen_his_1 = next(gen1)
                stop = False
            except StopIteration:
                pass
            try:
                gen_his_2 = next(gen2)
                stop = False
            except StopIteration:
                pass
            yield gen_his_1, gen_his_2
            if stop:
                break


def bot_response(history, handler, test_entry, inference_data, model_target):
    
    global inference_data_1, inference_data_2
    
    for item in handler.inference(test_entry, inference_data):
        # Processing logic remains the same
        if item[0] == "regular":
            responses_results = equalize_and_zip(item[1], item[2])
            for (model_res, exec_res) in responses_results:
                if model_res is not None:
                    response = model_res
                    history.append({"role": "assistant", "content": "Model Response: "})
                    for character in response:
                        history[-1]["content"] += character
                        yield history
                        time.sleep(0.01)
                        
                if exec_res is not None:
                    response = exec_res
                    history.append({"role": "assistant", "content": "<span class='bot-highlight'> Model Execution: </span>"})
                    for character in response:
                        history[-1]["content"] = history[-1]["content"][0:-7] + character + "</span>"
                        yield history
                        time.sleep(0.01)
                        
        elif item[0] == 'summary':
            response = item[1]
            if response is not None:
                history.append({"role": "assistant", "content": ""})
                for character in response:
                    history[-1]["content"] += character
                    yield history
                    time.sleep(0.01)
                    
        elif item[0] == "final":
            # Update inference data based on the target
            if model_target == "Model 1":
                inference_data_1 = item[2]
            elif model_target == "Model 2":
                inference_data_2 = item[2]
                

# Function to assign new unique ID on restart or at the start
def restart_chat_1(history):
    if len(history) > 2:  
        document = {"model": current_model_1,
                    "category": current_category_1,
                    "temperature": temperature_slider_1.value,
                    "history": history}
        collection.insert_one(document)
    global test_entry_1
    test_entry_1["id"] = str(uuid.uuid4())  # Reinitialize test_entry with new ID
    #update_handler_1(model_dropdown_1.value, temperature_slider_1.value)
    # print("test: entry", test_entry_1)
    return [{"role": "user", "content": "Hi, can you help me with some tasks?"}, 
            {"role": "assistant", "content": "Hello there! How can I assist you today?"}]

def restart_chat_2(history):
    if len(history) > 2:  
        document = {"model": current_model_2,
                    "category": current_category_2,
                    "temperature": temperature_slider_2.value,
                    "history": history}
        collection.insert_one(document)
    global test_entry_2
    test_entry_2["id"] = str(uuid.uuid4())  # Reinitialize test_entry with new ID
    #update_handler_2(model_dropdown_2.value, temperature_slider_2.value)
    # print("test: entry", test_entry_2)
    return [{"role": "user", "content": "Hi, can you help me with some tasks?"}, 
            {"role": "assistant", "content": "Hello there! How can I assist you today?"}]

# Function to report an issue
def report_issue():
    return gr.Info("Thank you for reporting the issue. Our team will look into it.")

# Update handler 1 when model or temperature is changed
def update_handler_1(model, temp_slider, history):
    print("update handler 1: ", model, temp_slider)
    global handler_1
    global current_model_1
    handler_1 = get_handler(model, temp_slider)  # Reinitialize handler with new model and temperature
    restart_history = restart_chat_1(history)
    current_model_1 = model
    return model, restart_history

# Update handler 2 when model or temperature is changed
def update_handler_2(model, temp_slider, history):
    print("update handler 2: ", model, temp_slider)
    global handler_2
    global current_model_2
    handler_2 = get_handler(model, temp_slider)  # Reinitialize handler with new model and temperature
    restart_history = restart_chat_2(history)
    current_model_2 = model
    return model, restart_history

# Update involved_classes and load config based on category
def update_category_and_load_config(category):
    global test_entry_1, test_entry_2
    
    print("update category: ", category)
    global current_category_1
    global current_category_2
    current_category_1 = category
    current_category_2 = category

    # Update involved_classes
    test_entry_1["initial_config"] = {category: {}}
    test_entry_1["involved_classes"] = [category]
    
    test_entry_2["initial_config"] = {category: {}}
    test_entry_2["involved_classes"] = [category]

    # Load the JSON file from the config folder corresponding to the category
    config_path = os.path.join("config", f"{category}.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as config_file:
            data = json.load(config_file)
            test_entry_1["function"] = data.copy()  # Load JSON content into test_entry["function"]
            test_entry_2["function"] = data.copy()
    
    return category

def load_example(example):
    if example == "Example 1 - GFSFileSystem":
        return models[0], 0.8, models[1], 0.3, categories[0], "Move final_report.pdf' within document directory to 'temp' directory in document. Make sure to create the directory"
    elif example == "Example 2 - TradingBot":
        return models[1], 0.9, models[2], 0.7, categories[5], "I'm contemplating enhancing my investment portfolio with some tech industry assets, and I've got my eye on Nvidia Corp. I'm keen to know its current stock price, and would appreciate if you could source this information for me."
    elif example == "Example 3 - TravelAPI":
        return models[2], 0.7, models[0], 0.2, categories[6], "As I plan a getaway, I'm curious about all the airports available for my travel. Would you share that information with me?"
    return models[0], 0.7, models[2], 0.36, categories[0], "Move final_report.pdf' within document directory to 'temp' directory in document. Make sure to create the directory"

# Add logic to load examples when example buttons are clicked
def load_example_and_update(example):
    # Load the example configuration
    model_1, temp_1, model_2, temp_2, category, message = load_example(example)
    update_category_and_load_config(category)
    # Update the interface components
    return model_1, temp_1, model_2, temp_2, category, message

# initialize test entry with default configurations
update_category_and_load_config(categories[0])


with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Multiturn LLM Chat Interface")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Configurations")

            model_dropdown_1 = gr.Dropdown(choices=models, label="Select Model 1", value=DEFAULT_MODEL_1, interactive=True)
            temperature_slider_1 = gr.Slider(0, 1, value=DEFAULT_TEMPERATURE_1, label="Temperature 1", interactive=True)
            
            model_dropdown_2 = gr.Dropdown(choices=models, label="Select Model 2", value=DEFAULT_MODEL_2, interactive=True)
            temperature_slider_2 = gr.Slider(0, 1, value=DEFAULT_TEMPERATURE_2, label="Temperature 2", interactive=True)
            
            category_dropdown = gr.Dropdown(choices=categories, label="Select Category", value=categories[0], interactive=True)
        
        with gr.Column(scale=3):
                
            with gr.Row():
                
                with gr.Column():
                    gr.Markdown("### Model 1")
                    chatbot1 = gr.Chatbot(elem_id="chatbot1", bubble_full_width=False, type="messages")

                    with gr.Row():
                        restart_btn_1 = gr.Button("Restart")
                        report_btn_1 = gr.Button("Report Issue")
                        
                with gr.Column():
                    gr.Markdown("### Model 2")
                    chatbot2 = gr.Chatbot(elem_id="chatbot2", bubble_full_width=False, type="messages")

                    with gr.Row():
                        restart_btn_2 = gr.Button("Restart")
                        report_btn_2 = gr.Button("Report Issue")
            with gr.Row():
                
                target_dropdown = gr.Dropdown(choices=["Model 1", "Model 2", "Both"], container=False, value="Both", interactive=True, scale=1)      
                chat_input = gr.MultimodalTextbox(
                    interactive=True,
                    file_count="multiple",
                    placeholder="Enter message or upload file...",
                    show_label=False,
                    scale=4
                )

            demo.load(lambda: get_initial_state(), outputs=chatbot1)
            demo.load(lambda: get_initial_state(), outputs=chatbot2)
            
            chat_msg = chat_input.submit(
                add_message, [chatbot1, chatbot2, chat_input, target_dropdown], [chatbot1, chatbot2, chat_input]
            ) 

            bot_msg = chat_msg.then(bot, inputs=[chatbot1, chatbot2, target_dropdown], outputs=[chatbot1, chatbot2])
            
            bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

            chatbot1.like(print_like_dislike, None, None, like_user_message=True)
            chatbot2.like(print_like_dislike, None, None, like_user_message=True)
            
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
    model_dropdown_1.change(
        update_handler_1, 
        [model_dropdown_1, temperature_slider_1, chatbot1], 
        [model_dropdown_1, chatbot1]
    )
    
    # Update handler when the model or temperature is changed
    model_dropdown_2.change(
        update_handler_2, 
        [model_dropdown_2, temperature_slider_2, chatbot2], 
        [model_dropdown_2, chatbot2]
    )

    # Update category and load config when a category is selected
    category_dropdown.change(
        update_category_and_load_config,
        inputs=category_dropdown,
        outputs=category_dropdown
    )

    # Set up the event handler for the restart button to reset the chat and test_entry
    restart_btn_1.click(restart_chat_1, [chatbot1], [chatbot1])
    report_btn_1.click(report_issue, None, None)
    
    restart_btn_2.click(restart_chat_2, [chatbot2], [chatbot2])
    report_btn_2.click(report_issue, None, None)
    
    example_btn1.click(load_example_and_update, inputs=example_btn1, 
                       outputs=[model_dropdown_1, temperature_slider_1, model_dropdown_2, temperature_slider_2, category_dropdown, chat_input])
    example_btn2.click(load_example_and_update, inputs=example_btn2, 
                       outputs=[model_dropdown_1, temperature_slider_1, model_dropdown_2, temperature_slider_2, category_dropdown, chat_input])
    example_btn3.click(load_example_and_update, inputs=example_btn3, 
                       outputs=[model_dropdown_1, temperature_slider_1, model_dropdown_2, temperature_slider_2, category_dropdown, chat_input])

# print("temperature_slider_1: ", temperature_slider_1.value)
# print("temperature_slider_2: ", temperature_slider_2.value)
demo.launch(share=True)





