"""
OpenRouter Responses API Configuration.

OpenRouter supports the Responses API at https://openrouter.ai/api/v1/responses
with OpenAI-compatible request/response format, including reasoning with
encrypted_content for multi-turn stateless workflows.

Docs: https://openrouter.ai/docs/api/reference/responses/overview
"""

from typing import Any, Optional

import httpx

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.responses.main import ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class OpenRouterResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for OpenRouter's Responses API.

    Inherits from OpenAIResponsesAPIConfig since OpenRouter's Responses API
    is compatible with OpenAI's Responses API specification.

    Key difference from direct OpenAI:
    - Uses https://openrouter.ai/api/v1 as the API base
    - Uses OPENROUTER_API_KEY for authentication
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OPENROUTER

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or litellm.api_key
            or get_secret_str("OPENROUTER_API_KEY")
            or get_secret_str("OR_API_KEY")
        )

        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY "
                "environment variable or pass api_key parameter."
            )

        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = (
            api_base
            or litellm.api_base
            or get_secret_str("OPENROUTER_API_BASE")
            or "https://openrouter.ai/api/v1"
        )

        api_base = api_base.rstrip("/")

        return f"{api_base}/responses"

    def transform_responses_api_request(
        self,
        model: str,
        input,
        response_api_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        request = super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Always ask OpenRouter to include authoritative per-request cost data.
        if "usage" not in request:
            request["usage"] = {"include": True}

        return request

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> ResponsesAPIResponse:
        response = super().transform_response_api_response(
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        # OpenRouter returns cost information in usage when usage.include=true.
        try:
            response_json = raw_response.json()
            if "usage" in response_json and response_json["usage"]:
                response_cost = response_json["usage"].get("cost")
                if response_cost is not None:
                    if not hasattr(response, "_hidden_params"):
                        response._hidden_params = {}
                    if "additional_headers" not in response._hidden_params:
                        response._hidden_params["additional_headers"] = {}
                    response._hidden_params["additional_headers"][
                        "llm_provider-x-litellm-response-cost"
                    ] = float(response_cost)
        except Exception:
            # If we can't extract cost, continue without it - don't fail the response.
            pass

        return response

    def supports_native_websocket(self) -> bool:
        """OpenRouter does not support native WebSocket for Responses API"""
        return False
