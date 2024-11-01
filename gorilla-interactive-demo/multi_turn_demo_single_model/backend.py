from openai_helper import OpenAIHandler

def get_handler(model_name, temperature):
    return OpenAIHandler(model_name, temperature)