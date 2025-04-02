import pytest
from unittest.mock import MagicMock
import json
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
import litellm
from litellm import ModelResponse

def test_transform_response_with_avglogprobs():
    # Create a mock response with avgLogprobs
    response_json = {
        "candidates": [{
            "content": {"parts": [{"text": "Test response"}], "role": "model"},
            "finishReason": "STOP",
            "avgLogprobs": -0.3445799010140555
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15
        }
    }
    
    # Create mock objects
    mock_response = MagicMock()
    mock_response.text = json.dumps(response_json)
    mock_response.json.return_value = response_json
    mock_response.status_code = 200
    mock_response.headers = {}
    
    # Mock logging object
    mock_logging_obj = MagicMock()
    
    # Create a model response object
    model_response = ModelResponse()
    # Initialize with an empty choice to avoid IndexError
    model_response.choices = [litellm.Choices(finish_reason=None, index=0, message={})]
    
    # Call transform_response
    config = VertexGeminiConfig()
    transformed_response = config.transform_response(
        model='gemini-2.0-flash',
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=mock_logging_obj,
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
        api_key=None
    )
    
    # Assert that the avgLogprobs was correctly added to the model response
    # The transform_response method adds a new choice with the avgLogprobs
    assert len(transformed_response.choices) == 2
    # Check that the second choice has the avgLogprobs
    assert transformed_response.choices[1].logprobs == -0.3445799010140555