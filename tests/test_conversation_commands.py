from types import SimpleNamespace

import pytest

from astrbot.builtin_stars.builtin_commands.commands import (
    conversation as conversation_module,
)


@pytest.mark.asyncio
async def test_clear_third_party_agent_runner_state_deletes_deerflow_thread_before_local_state(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[object] = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        async def delete_thread(self, thread_id: str, timeout: float = 20):
            calls.append(("delete", thread_id, timeout))

        async def close(self):
            calls.append(("close",))

    async def fake_get_async(*args, **kwargs):
        _ = args, kwargs
        return "thread-123"

    async def fake_remove_async(*args, **kwargs):
        calls.append(("remove", kwargs["scope"], kwargs["scope_id"], kwargs["key"]))

    context = SimpleNamespace(
        get_config=lambda **kwargs: {
            "provider_settings": {"deerflow_agent_runner_provider_id": "deerflow-runner"}
        },
        provider_manager=SimpleNamespace(
            get_provider_config_by_id=lambda provider_id, merged=False: {
                "id": provider_id,
                "deerflow_api_base": "http://127.0.0.1:2026",
                "deerflow_api_key": "token",
                "deerflow_auth_header": "",
                "proxy": "",
            }
            if merged
            else {"id": provider_id},
        ),
    )

    monkeypatch.setattr(conversation_module, "DeerFlowAPIClient", FakeClient)
    monkeypatch.setattr(conversation_module.sp, "get_async", fake_get_async)
    monkeypatch.setattr(conversation_module.sp, "remove_async", fake_remove_async)

    await conversation_module._clear_third_party_agent_runner_state(
        context,
        "umo-1",
        conversation_module.DEERFLOW_PROVIDER_TYPE,
    )

    assert ("delete", "thread-123", 20) in calls
    assert (
        "remove",
        "umo",
        "umo-1",
        conversation_module.DEERFLOW_THREAD_ID_KEY,
    ) in calls
    assert calls.index(("delete", "thread-123", 20)) < calls.index(
        ("remove", "umo", "umo-1", conversation_module.DEERFLOW_THREAD_ID_KEY)
    )


@pytest.mark.asyncio
async def test_clear_third_party_agent_runner_state_removes_local_state_when_deerflow_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[object] = []

    class FakeClient:
        def __init__(self, **kwargs):
            _ = kwargs

        async def delete_thread(self, thread_id: str, timeout: float = 20):
            _ = thread_id, timeout
            raise RuntimeError("gateway down")

        async def close(self):
            calls.append(("close",))

    async def fake_get_async(*args, **kwargs):
        _ = args, kwargs
        return "thread-456"

    async def fake_remove_async(*args, **kwargs):
        calls.append(("remove", kwargs["scope"], kwargs["scope_id"], kwargs["key"]))

    context = SimpleNamespace(
        get_config=lambda **kwargs: {
            "provider_settings": {"deerflow_agent_runner_provider_id": "deerflow-runner"}
        },
        provider_manager=SimpleNamespace(
            get_provider_config_by_id=lambda provider_id, merged=False: {
                "id": provider_id,
                "deerflow_api_base": "http://127.0.0.1:2026",
                "deerflow_api_key": "",
                "deerflow_auth_header": "",
                "proxy": "",
            }
            if merged
            else {"id": provider_id},
        ),
    )

    monkeypatch.setattr(conversation_module, "DeerFlowAPIClient", FakeClient)
    monkeypatch.setattr(conversation_module.sp, "get_async", fake_get_async)
    monkeypatch.setattr(conversation_module.sp, "remove_async", fake_remove_async)

    await conversation_module._clear_third_party_agent_runner_state(
        context,
        "umo-2",
        conversation_module.DEERFLOW_PROVIDER_TYPE,
    )

    assert (
        "remove",
        "umo",
        "umo-2",
        conversation_module.DEERFLOW_THREAD_ID_KEY,
    ) in calls


@pytest.mark.asyncio
async def test_clear_third_party_agent_runner_state_removes_local_state_when_deerflow_client_init_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[object] = []

    class FakeClient:
        def __init__(self, **kwargs):
            _ = kwargs
            raise RuntimeError("invalid deerflow config")

    async def fake_get_async(*args, **kwargs):
        _ = args, kwargs
        return "thread-789"

    async def fake_remove_async(*args, **kwargs):
        calls.append(("remove", kwargs["scope"], kwargs["scope_id"], kwargs["key"]))

    context = SimpleNamespace(
        get_config=lambda **kwargs: {
            "provider_settings": {"deerflow_agent_runner_provider_id": "deerflow-runner"}
        },
        provider_manager=SimpleNamespace(
            get_provider_config_by_id=lambda provider_id, merged=False: {
                "id": provider_id,
                "deerflow_api_base": "http://127.0.0.1:2026",
                "deerflow_api_key": "",
                "deerflow_auth_header": "",
                "proxy": "",
            }
            if merged
            else {"id": provider_id},
        ),
    )

    monkeypatch.setattr(conversation_module, "DeerFlowAPIClient", FakeClient)
    monkeypatch.setattr(conversation_module.sp, "get_async", fake_get_async)
    monkeypatch.setattr(conversation_module.sp, "remove_async", fake_remove_async)

    await conversation_module._clear_third_party_agent_runner_state(
        context,
        "umo-3",
        conversation_module.DEERFLOW_PROVIDER_TYPE,
    )

    assert (
        "remove",
        "umo",
        "umo-3",
        conversation_module.DEERFLOW_THREAD_ID_KEY,
    ) in calls


# ════════════════════════════════════════
# /stats context-window display
# ════════════════════════════════════════


def _make_commands(provider_max_ctx: int = 0, model: str = "claude-sonnet-4-5",
                   fallback: int = 128000):
    """Construct ConversationCommands with a stub context for unit-testing
    pure helper methods (no DB / event involvement)."""
    provider = SimpleNamespace(
        provider_config={"max_context_tokens": provider_max_ctx},
        get_model=lambda: model,
        meta=lambda: SimpleNamespace(id="prov1"),
    )
    context = SimpleNamespace(
        get_using_provider=lambda umo=None: provider,
        get_config=lambda umo=None: {
            "provider_settings": {"fallback_max_context_tokens": fallback}
        },
    )
    return conversation_module.ConversationCommands(context)


def test_resolve_max_context_tokens_uses_explicit_config():
    cmds = _make_commands(provider_max_ctx=200000)
    provider = cmds.context.get_using_provider()
    assert cmds._resolve_max_context_tokens(provider, "umo") == 200000


def test_resolve_max_context_tokens_falls_back_to_metadata(monkeypatch):
    cmds = _make_commands(provider_max_ctx=0, model="meta-test-model")
    provider = cmds.context.get_using_provider()
    monkeypatch.setattr(
        "astrbot.core.utils.llm_metadata.LLM_METADATAS",
        {"meta-test-model": {"limit": {"context": 64000}}},
    )
    assert cmds._resolve_max_context_tokens(provider, "umo") == 64000


def test_resolve_max_context_tokens_uses_fallback_setting(monkeypatch):
    cmds = _make_commands(provider_max_ctx=0, model="unknown-model", fallback=99999)
    provider = cmds.context.get_using_provider()
    monkeypatch.setattr("astrbot.core.utils.llm_metadata.LLM_METADATAS", {})
    assert cmds._resolve_max_context_tokens(provider, "umo") == 99999


def test_format_context_line_unknown_max():
    cmds = _make_commands()
    line = cmds._format_context_line(0, 0)
    assert "unknown" in line


def test_format_context_line_no_current_tokens_yet():
    cmds = _make_commands()
    line = cmds._format_context_line(0, 200_000)
    # Compaction threshold = 82% of 200k = 164,000
    assert "164,000" in line
    assert "200,000" in line


def test_format_context_line_under_threshold():
    cmds = _make_commands()
    line = cmds._format_context_line(50_000, 200_000)
    # remaining until compact = 164000 - 50000 = 114000
    assert "50,000" in line
    assert "114,000" in line
    assert "until compact" in line


def test_format_context_line_over_threshold():
    cmds = _make_commands()
    # 180k > 164k → over threshold by 16k
    line = cmds._format_context_line(180_000, 200_000)
    assert "180,000" in line
    assert "16,000" in line
    assert "over compact threshold" in line
