# src/app/endpoints/chat.py
import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.logger import logger
from app.openapi.chat_completions import (
    CHAT_COMPLETIONS_REQUEST_EXAMPLES,
    CHAT_COMPLETIONS_RESPONSE_200,
    TEMPORARY_CHAT_COMPLETIONS_REQUEST_EXAMPLES,
    TEMPORARY_CHAT_COMPLETIONS_RESPONSE_400,
)
from app.schemas.request import GeminiRequest, OpenAIChatRequest
from app.services.gemini_client import get_gemini_client, GeminiClientNotInitializedError
from app.services.providers.gemini.session_manager import get_translate_session_manager
from app.services.factory import ProviderFactory
from app.services.model_catalog import list_models as build_model_catalog
from app.services.providers.gemini.temporary_chat import handle_temporary_chat_completions

router = APIRouter()


@router.get(
    "/v1/gems",
    tags=["Utilities"],
    summary="List Available Gems",
    description="Returns available Gemini Gems associated with the account. Can be used to apply specific personas in chat requests."
)
async def list_gems():
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        gems = await gemini_client.fetch_gems()
        return {
            "gems": [
                {
                    "id": gem.id,
                    "name": gem.name,
                    "description": gem.description,
                    "predefined": gem.predefined,
                }
                for gem in gems
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching gems: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching gems: {str(e)}")


@router.post(
    "/translate",
    tags=["Translation"],
    summary="Translate Extension Compatibility",
    description="Extension-specific translation endpoint retained for compatibility with Translate It!-style browser extensions. This endpoint uses a shared global in-memory session, sends Gemini WebAPI translation requests as temporary requests so they are not saved in Gemini history, has no `conversation_id` support, does not support streaming, and does not survive server restarts. The client is responsible for sending a translation-specific prompt. For isolated or persistent translation workflows, use `/v1/chat/completions`."
)
async def translate_chat(request: GeminiRequest):
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    session_manager = get_translate_session_manager()
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager is not initialized.")
    try:
        response = await session_manager.get_response(
            request.model,
            request.message,
            request.files,
            request.gem,
            temporary=True,
        )
        return {"response": response.text}
    except Exception as e:
        logger.error(f"Error in /translate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during translation: {str(e)}")


@router.post(
    "/v1/temporary/chat/completions",
    tags=["Chat"],
    summary="Temporary OpenAI-Compatible Chat Completions",
    description=(
        "Gemini WebAPI-only OpenAI-compatible chat completions endpoint. Requests are sent with temporary=True, "
        "so responses are not saved in Gemini history and do not write SQLite conversation snapshots. "
        "`conversation_id` is rejected. Playwright models/providers, Atlas models/providers, and any non-Gemini provider are rejected. "
        "The endpoint supports streaming and non-streaming responses. File content parts are supported only by "
        "Gemini WebAPI, are request-scoped, and generated artifact metadata follows the same response shape as "
        "`/v1/chat/completions`."
    ),
    responses={
        200: CHAT_COMPLETIONS_RESPONSE_200,
        400: TEMPORARY_CHAT_COMPLETIONS_RESPONSE_400,
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": TEMPORARY_CHAT_COMPLETIONS_REQUEST_EXAMPLES,
                }
            }
        }
    },
)
async def temporary_chat_completions(request: OpenAIChatRequest):
    return await handle_temporary_chat_completions(request)


@router.get(
    "/v1/models",
    tags=["Chat"],
    summary="List Available Models",
    description="Returns available models from all registered providers. Includes provider-prefixed models used for discovery and routing."
)
async def get_models():
    return await build_model_catalog(include_legacy_playwright_aliases=False, allow_stale=False)


@router.post(
    "/v1/responses",
    tags=["Chat"],
    summary="Codex Responses API Compatibility",
    description="Endpoint to support the newer /v1/responses protocol used by Codex by seamlessly translating it to internal /v1/chat/completions logic."
)
async def responses_api(raw_request: Request):
    try:
        payload = await raw_request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # 1. 将 Responses 协议中的 input 转换为内部兼容的 messages
    if "input" in payload:
        payload["messages"] = payload.pop("input")
    
    # 2. 修复数据结构差异：Codex 使用 "input_text"，而 OpenAI 规范使用 "text"
    for msg in payload.get("messages", []):
        if isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if isinstance(part, dict):
                    # 将 input_text 映射为 text
                    if part.get("type") == "input_text":
                        part["type"] = "text"
                    # 如果后续 Codex 传文件/图片报错，也可以在这里添加映射规则，例如：
                    # elif part.get("type") == "input_image": ...

    # 移除可能引起 Pydantic 校验错误的特有参数
    payload.pop("context_management", None)

    try:
        # 使用转换后的 payload 实例化标准的 OpenAIChatRequest
        chat_req = OpenAIChatRequest(**payload)
    except Exception as e:
        # 如果再次报错，错误信息会在这里被捕获
        raise HTTPException(status_code=400, detail=f"Request parsing error: {e}")

    # 附加 http_request_id (与原 chat_completions 保持一致)
    if hasattr(raw_request.state, "request_id"):
        object.__setattr__(chat_req, "_http_request_id", raw_request.state.request_id)

    # 解析 provider
    provider, resolved_model = ProviderFactory.get_provider(chat_req)
    chat_req.model = resolved_model

    # 3. 获取 Provider 的标准响应
    original_response = await provider.chat_completions(chat_req)

    # 4. 拦截并转换流式输出
    if isinstance(original_response, StreamingResponse):
        async def response_stream_generator():
            buffer = ""
            async for chunk in original_response.body_iterator:
                if isinstance(chunk, bytes):
                    buffer += chunk.decode("utf-8")
                else:
                    buffer += chunk
                
                # 按行处理 SSE 数据
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            continue
                        
                        try:
                            data_json = json.loads(data_str)
                            # 提取 delta 重新封装为 output
                            if "choices" in data_json and len(data_json["choices"]) > 0:
                                delta = data_json["choices"][0].get("delta", {})
                                response_chunk = {
                                    "id": data_json.get("id", "resp-id"),
                                    "object": "response.chunk",
                                    "output": [delta]
                                }
                                yield f"data: {json.dumps(response_chunk)}\n\n"
                        except json.JSONDecodeError:
                            continue
                            
        return StreamingResponse(response_stream_generator(), media_type="text/event-stream")
    
    # 5. 拦截并转换非流式输出
    else:
        if hasattr(original_response, "model_dump"):
            data = original_response.model_dump()
        elif hasattr(original_response, "dict"):
            data = original_response.dict()
        else:
            data = original_response

        response_data = {
            "id": data.get("id", "resp-123"),
            "object": "response",
            "output": [choice["message"] for choice in data.get("choices", [])]
        }
        return response_data


@router.post(
    "/v1/chat/completions",
    tags=["Chat"],
    summary="OpenAI-Compatible Chat Completions",
    description=(
        "Primary OpenAI-compatible chat completions endpoint. Gemini WebAPI supports file content parts; file parts are request-scoped and unsupported backends reject them. "
        "For Gemini WebAPI, text parts are concatenated into one prompt and file parts are passed as attachments, so exact text/file interleaving is not preserved. "
        "Supported file formats are documented in docs/api.md. This is the recommended API for new integrations."
    ),
    responses={200: CHAT_COMPLETIONS_RESPONSE_200},
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": CHAT_COMPLETIONS_REQUEST_EXAMPLES,
                }
            }
        }
    },
)
async def chat_completions(request: OpenAIChatRequest, http_request: Request):
    # Attach HTTP request_id for observability (will be used by adapter if present)
    # The middleware sets request.state.request_id
    if hasattr(http_request.state, "request_id"):
        # Attach to the Pydantic model as an extra attribute (not validated).
        # NOTE: This is for observability only and NOT part of the API contract.
        # Clients should NOT rely on this field.
        object.__setattr__(request, "_http_request_id", http_request.state.request_id)

    # Resolve provider and model name via the static factory
    provider, resolved_model = ProviderFactory.get_provider(request)

    # Update the request with the resolved model name so the provider gets the clean version
    request.model = resolved_model

    # Delegate implementation-heavy work to the provider
    return await provider.chat_completions(request)


@router.get(
    "/v1/conversations",
    tags=["Chat"],
    summary="List Gemini WebAPI Conversations",
    description="Lists locally persisted Gemini WebAPI conversations stored in SQLite. Playwright and Atlas conversations are not included."
)
async def list_conversations():
    provider, _ = ProviderFactory.get_provider(
        OpenAIChatRequest(messages=[], provider="gemini")
    )
    list_handler = getattr(provider, "list_conversations", None)
    if list_handler is None:
        raise HTTPException(status_code=400, detail="Conversation listing is not supported for this provider.")
    return await list_handler()


@router.delete(
    "/v1/conversations",
    tags=["Chat"],
    summary="Bulk Delete Gemini WebAPI Conversations",
    description="Deletes all locally persisted Gemini WebAPI conversations. Playwright and Atlas conversations are not supported."
)
async def delete_conversations():
    provider, _ = ProviderFactory.get_provider(
        OpenAIChatRequest(messages=[], provider="gemini")
    )
    delete_handler = getattr(provider, "delete_conversations", None)
    if delete_handler is None:
        raise HTTPException(status_code=400, detail="Bulk conversation deletion is not supported for this provider.")
    return await delete_handler()


@router.delete(
    "/v1/conversations/{conversation_id}",
    tags=["Chat"],
    summary="Delete Gemini WebAPI Conversation",
    description="Deletes a Gemini WebAPI conversation by local conversation_id. Playwright and Atlas conversations are not supported."
)
async def delete_conversation(conversation_id: str):
    provider, _ = ProviderFactory.get_provider(
        OpenAIChatRequest(messages=[], provider="gemini")
    )
    delete_handler = getattr(provider, "delete_conversation", None)
    if delete_handler is None:
        raise HTTPException(status_code=400, detail="Conversation deletion is not supported for this provider.")
    return await delete_handler(conversation_id)