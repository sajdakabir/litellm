import json
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import cohere_messages_pt_v2
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage

from ..common_utils import ModelResponseIterator as CohereModelResponseIterator
from ..common_utils import validate_environment as cohere_validate_environment

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class CohereError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[httpx.Headers] = None,
    ):
        self.status_code = status_code
        self.message = message
        # Updated URL to v2 endpoint
        self.request = httpx.Request(method="POST", url="https://api.cohere.ai/v2/chat")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


class CohereChatConfigV2(BaseConfig):
    """
    Configuration class for Cohere's API v2 interface.

    Args:
        temperature (float, optional): A non-negative float that tunes the degree of randomness in generation.
        max_tokens (int, optional): The maximum number of tokens the model will generate as part of the response.
        max_completion_tokens (int, optional): The maximum number of tokens the model will generate as part of the response.
        k (int, optional): Ensures only the top k most likely tokens are considered for generation at each step.
        p (float, optional): Ensures that only the most likely tokens, with total probability mass of p, are considered for generation.
        frequency_penalty (float, optional): Used to reduce repetitiveness of generated tokens.
        presence_penalty (float, optional): Used to reduce repetitiveness of generated tokens.
        tools (List[Dict[str, str]], optional): A list of available tools (functions) that the model may suggest invoking.
        seed (int, optional): A seed to assist reproducibility of the model's response.
        documents (List[Dict[str, str]], optional): A list of relevant documents that the model can cite.
        stop_sequences (List[str], optional): A list of sequences where the API will stop generating further tokens.
        stream (bool, optional): Whether to stream the response or not.
    """

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    k: Optional[int] = None
    p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    tools: Optional[List[Dict[str, Any]]] = None
    seed: Optional[int] = None
    documents: Optional[List[Dict[str, str]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = None

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        k: Optional[int] = None,
        p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        documents: Optional[List[Dict[str, str]]] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: Optional[bool] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        # Use the common validation function but specify v2 API base if not provided
        if api_base is None:
            api_base = "https://api.cohere.ai/v2/chat"
            
        return cohere_validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
        )

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "stream",
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "tools",
            "tool_choice",
            "seed",
            "documents",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "stream":
                optional_params["stream"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            if param == "n":
                optional_params["num_generations"] = value
            if param == "top_p":
                optional_params["p"] = value
            if param == "frequency_penalty":
                optional_params["frequency_penalty"] = value
            if param == "presence_penalty":
                optional_params["presence_penalty"] = value
            if param == "stop":
                optional_params["stop_sequences"] = value
            if param == "tools":
                optional_params["tools"] = value
            if param == "seed":
                optional_params["seed"] = value
            if param == "documents":
                optional_params["documents"] = value
        return optional_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        ## Load Config
        for k, v in litellm.CohereChatConfigV2.get_config().items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > cohere_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        # In v2, we need to provide the messages in a specific format
        # Convert OpenAI format messages to Cohere v2 format
        cohere_messages = []
        
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            
            # Map OpenAI roles to Cohere v2 roles
            if role == "system":
                cohere_messages.append({"role": "system", "message": content})
            elif role == "user":
                cohere_messages.append({"role": "user", "message": content})
            elif role == "assistant":
                # Handle the case where content might be a list of content blocks
                if isinstance(content, list):
                    # For simplicity, we're just extracting text content
                    text_content = ""
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content += item.get("text", "")
                    cohere_messages.append({"role": "assistant", "message": text_content})
                else:
                    cohere_messages.append({"role": "assistant", "message": content})
            elif role == "tool":
                # Handle tool responses
                tool_call_id = message.get("tool_call_id")
                name = message.get("name")
                content = message.get("content")
                
                # In v2, tool responses are handled differently
                cohere_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "message": content
                })
        
        # Add messages to optional_params
        optional_params["message"] = cohere_messages[-1]["message"] if cohere_messages else ""
        optional_params["chat_history"] = cohere_messages[:-1] if len(cohere_messages) > 1 else []

        ## Handle Tool Calling
        if "tools" in optional_params:
            cohere_tools = self._construct_cohere_tool(tools=optional_params["tools"])
            optional_params["tools"] = cohere_tools
            
        # Handle tool_choice parameter
        if "tool_choice" in optional_params:
            tool_choice = optional_params.pop("tool_choice")
            if tool_choice == "auto":
                # In Cohere v2, this is the default behavior
                pass
            elif tool_choice == "none":
                # Disable tool calling
                optional_params["tools"] = []
            elif isinstance(tool_choice, dict) and "function" in tool_choice:
                # Specify a specific tool to use
                function_name = tool_choice["function"].get("name")
                if function_name and "tools" in optional_params:
                    # Filter tools to only include the specified function
                    optional_params["tools"] = [
                        tool for tool in optional_params["tools"]
                        if tool.get("function", {}).get("name") == function_name
                    ]

        return optional_params

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        try:
            raw_response_json = raw_response.json()
            
            # Handle the response format from Cohere v2 API
            if "text" in raw_response_json:
                # This is the standard v2 response format
                model_response.choices[0].message.content = raw_response_json["text"]  # type: ignore
                
                # Handle tool calls if present
                if "tool_calls" in raw_response_json and len(raw_response_json["tool_calls"]) > 0:
                    tool_calls = []
                    for idx, tool_call in enumerate(raw_response_json["tool_calls"]):
                        # Generate a unique ID for each tool call if not provided
                        tool_id = tool_call.get("id", f"call_{idx}")
                        
                        tool_calls.append({
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["parameters"])
                            }
                        })
                    model_response.choices[0].message.tool_calls = tool_calls  # type: ignore
                    # If there are tool calls, set the finish reason accordingly
                    model_response.choices[0].finish_reason = "tool_calls"  # type: ignore
                
                # Handle citations if present
                if "citations" in raw_response_json and len(raw_response_json["citations"]) > 0:
                    model_response.citations = raw_response_json["citations"]
            else:
                # Handle other response formats or raise an error
                raise CohereError(
                    message="Invalid response format from Cohere API v2", 
                    status_code=raw_response.status_code
                )
        except Exception as e:
            raise CohereError(
                message=f"{raw_response.text} - Error: {str(e)}", status_code=raw_response.status_code
            )

        ## ADD CITATIONS
        if "citations" in raw_response_json:
            setattr(model_response, "citations", raw_response_json["citations"])

        ## CALCULATING USAGE - use cohere `billed_units` for returning usage
        billed_units = raw_response_json.get("meta", {}).get("billed_units", {})

        # Extract token usage information
        prompt_tokens = billed_units.get("input_tokens", 0)
        completion_tokens = billed_units.get("output_tokens", 0)

        # Set model response metadata
        model_response.id = raw_response_json.get("generation_id", f"gen_{int(time.time())}")
        model_response.created = int(time.time())
        model_response.model = model
        
        # Set usage information
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        
        return model_response

    def _construct_cohere_tool(
        self,
        tools: Optional[list] = None,
    ):
        if tools is None:
            tools = []
        cohere_tools = []
        for tool in tools:
            cohere_tool = self._translate_openai_tool_to_cohere(tool)
            cohere_tools.append(cohere_tool)
        return cohere_tools

    def _translate_openai_tool_to_cohere(
        self,
        openai_tool: dict,
    ):
        # In v2, Cohere uses JSON schema format similar to OpenAI
        # We can mostly pass through the OpenAI tool format with minor adjustments
        if "function" not in openai_tool:
            return openai_tool
            
        return {
            "name": openai_tool["function"]["name"],
            "description": openai_tool["function"].get("description", ""),
            "parameter_definitions": openai_tool["function"]["parameters"],
        }

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        # Create a custom iterator for v2 streaming
        return CohereV2ModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return CohereError(status_code=status_code, message=error_message)


class CohereV2ModelResponseIterator(CohereModelResponseIterator):
    """
    Custom iterator for Cohere API v2 streaming responses.
    """
    
    def __iter__(self):
        for chunk in self.streaming_response:
            if not chunk:
                continue
            
            try:
                # Parse the chunk if it's a string (from streaming response)
                if isinstance(chunk, str) or isinstance(chunk, bytes):
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode('utf-8')
                    chunk_data = json.loads(chunk)
                else:
                    chunk_data = chunk
                
                # Handle text chunks
                if "text" in chunk_data:
                    text = chunk_data.get("text", "")
                    if text:
                        yield {"choices": [{"delta": {"content": text}}]}
                
                # Handle tool call chunks
                if "tool_calls" in chunk_data and chunk_data["tool_calls"]:
                    tool_calls = []
                    for idx, tool_call in enumerate(chunk_data["tool_calls"]):
                        tool_id = tool_call.get("id", f"call_{idx}")
                        tool_calls.append({
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["parameters"])
                            }
                        })
                    
                    if tool_calls:
                        yield {"choices": [{"delta": {"tool_calls": tool_calls}}]}
                        
                # Handle citations if present
                if "citations" in chunk_data and chunk_data["citations"]:
                    yield {"citations": chunk_data["citations"]}
                    
            except Exception as e:
                # If we can't parse the chunk, just skip it
                continue
