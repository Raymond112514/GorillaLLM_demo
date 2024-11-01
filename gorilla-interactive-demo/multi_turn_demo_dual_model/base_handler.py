import json
import time

from multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from constant import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
    MAXIMUM_ROUND_LIMIT,
)
from model_style import ModelStyle


class BaseHandler:
    model_name: str
    model_style: ModelStyle

    def __init__(self, model_name, temperature) -> None:
        self.model_name = model_name
        # Replace the slash with underscore to avoid creating subdirectories
        # Replace the dash and dot with underscore for valid variable name
        self.model_name_underline_replaced = (
            model_name.replace("/", "_").replace("-", "_").replace(".", "_")
        )
        self.temperature = temperature
        self.is_fc_model = False  # Whether the model is a function calling model

    def inference(self, test_entry: dict, include_debugging_log: bool):
        # This method is used to retrive model response for each model.

        # FC model
        return self.inference_multi_turn_FC(test_entry, include_debugging_log)


    def inference_multi_turn_FC(
        self, test_entry: dict, inference_data: dict
    ) -> tuple[list[list], dict]:
        initial_config: dict = test_entry["initial_config"]
        involved_classes: list = test_entry["involved_classes"]
        test_entry_id: str = test_entry["id"]  # TODO: @Sharon, provide this, unique to each chat session

        inference_data = self._compile_tools(inference_data, test_entry)

        current_round_message: list[dict] = test_entry["question"]  # @Sharon, this should be list of one dict.

        inference_data = self._add_next_turn_user_message_FC(
            inference_data, current_round_message
        )

        current_round_response = []
        involved_instances = []
        count = 0
        while True:

            api_response = self._query_FC(inference_data)

            # Try parsing the model response
            model_response_data = self._parse_query_response_FC(api_response)
            model_responses = model_response_data["model_responses"]

            # Add the assistant message to the chat history
            inference_data = self._add_assistant_message_FC(
                inference_data, model_response_data
            )

            # Try decoding the model response
            try:
                decoded_model_responses = self.decode_execute(model_responses)

                if is_empty_execute_response(decoded_model_responses):
                    print("Empty response from the model. Proceed to next turn.")

                    break

            except Exception as e:
                print("Failed to decode the model response. Proceed to next turn.")
                # last step with only the model response
                yield ("summary", model_responses, None, self.model_name)
                break

            finally:
                current_round_response.append(model_responses)

            # Obtain the execution results
            execution_results, involved_instances = execute_multi_turn_func_call(
                decoded_model_responses,
                initial_config,
                involved_classes,
                self.model_name_underline_replaced,
                test_entry_id,
                long_context=False,
                is_evaL_run=False,
            )

            # Add the execution results to the chat history for the next round
            inference_data = self._add_execution_results_FC(
                inference_data, execution_results, model_response_data
            )

            yield ("regular", decoded_model_responses, execution_results, self.model_name)
            
            count += 1
            # Force quit after too many rounds
            if count > MAXIMUM_ROUND_LIMIT:
                print("Exceeded maximum round limit. Force quit.")
                break

        yield ("final", current_round_response, inference_data, involved_instances)


    def decode_ast(self, result, language="Python"):
        # This method takes raw model output and convert it to standard AST checker input.
        raise NotImplementedError

    def decode_execute(self, result):
        # This method takes raw model output and convert it to standard execute checker input.
        raise NotImplementedError


    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        """
        Call the model API in FC mode to get the response.
        Return the response object that can be used to feed into the decode method.
        """
        raise NotImplementedError

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        """
        Preprocess the testset entry before sending it to the model.
        This includes transforming the input user message into the format expected by the model, and any other necessary preprocessing steps.
        The inference_data dict is updated in place and returned.
        """
        raise NotImplementedError

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        """
        Compile the tools from the test entry and add them to the inference data.
        This method is used to prepare the tools for the model query in FC mode.
        The inference_data dict is updated in place and returned.
        """
        raise NotImplementedError

    def _parse_query_response_FC(self, api_response: any) -> dict:
        """
        Parses the raw response from the model API to extract the result, input token count, and output token count.

        Args:
            api_response (any): The raw response from the model API.

        Returns:
            A dict containing the following elements:
                - model_responses (any): The parsed result that can be directly used as input to the decode method.
                - input_token (int): The number of tokens used in the input to the model.
                - output_token (int): The number of tokens generated by the model as output.
                - tool_call_ids (list[str]): The IDs of the tool calls that are generated by the model. Optional.
                - Any other metadata that is specific to the model.
        """
        raise NotImplementedError

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """
        Add the first turn message to the chat history.
        """
        raise NotImplementedError

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """
        [Only for multi-turn]
        Add next round user message to the chat history for query.
        user_message is a list of 1 element, which is the user message.
        """
        raise NotImplementedError

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Add assistant message to the chat history.
        """
        raise NotImplementedError

    def _add_execution_results_FC(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next round of query.
        Some models may need to add additional information to the chat history, such as tool call IDs.
        """
        raise NotImplementedError

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        """
        Call the model API in prompting mode to get the response.
        Return the response object that can be used to feed into the decode method.
        """
        raise NotImplementedError

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        """
        Preprocess the testset entry before sending it to the model.
        Returns a dict that contains all the necessary information for the query method.
        `tools` and `message` must be included in the returned dict.
        Things like `system_prompt` and `chat_history` are optional, specific to the model.
        """
        raise NotImplementedError

    def _parse_query_response_prompting(self, api_response: any) -> dict:
        """
        Parses the raw response from the model API to extract the result, input token count, and output token count.

        Args:
            api_response (any): The raw response from the model API.

        Returns:
            A dict containing the following elements:
                - model_responses (any): The parsed result that can be directly used as input to the decode method.
                - input_token (int): The number of tokens used in the input to the model.
                - output_token (int): The number of tokens generated by the model as output.
                - tool_call_ids (list[str]): The IDs of the tool calls that are generated by the model. Optional.
                - Any other metadata that is specific to the model.
        """
        raise NotImplementedError

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """
        Add the first turn message to the chat history.
        """
        raise NotImplementedError

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """
        [Only for multi-turn]
        Add next round user message to the chat history for query.
        user_message is a list of 1 element, which is the user message.
        """
        raise NotImplementedError

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Add assistant message to the chat history.
        """
        raise NotImplementedError

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next round of query.
        Some models may need to add additional information to the chat history, such as tool call IDs.
        """
        raise NotImplementedError
