from doc_assistant.models.language_model import (
    AsyncChatModelProtocol,
    AsyncMessageChatModelProtocol,
    ChatModelProtocol,
    InvokableChatModelProtocol,
    MessageChatModelProtocol,
    MessageStreamingChatModelProtocol,
    StreamingChatModelProtocol,
)


class MessageOnlyChatModel:
    def invoke_messages(self, messages, tools=None, tool_choice="auto"):
        return {"content": "ok"}


class AsyncMessageOnlyChatModel:
    async def ainvoke_messages(self, messages, tools=None, tool_choice="auto"):
        return {"content": "ok"}


def test_message_only_model_matches_only_its_supported_capability() -> None:
    model = MessageOnlyChatModel()

    assert isinstance(model, MessageChatModelProtocol)
    assert not isinstance(model, InvokableChatModelProtocol)
    assert not isinstance(model, StreamingChatModelProtocol)
    assert not isinstance(model, MessageStreamingChatModelProtocol)
    assert not isinstance(model, ChatModelProtocol)


def test_async_message_only_model_does_not_require_generic_ainvoke() -> None:
    model = AsyncMessageOnlyChatModel()

    assert isinstance(model, AsyncMessageChatModelProtocol)
    assert not isinstance(model, AsyncChatModelProtocol)
