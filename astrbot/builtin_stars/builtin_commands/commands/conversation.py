from sqlalchemy import case, func, select
from sqlmodel import col

from astrbot.api import sp, star
from astrbot.api.event import AstrMessageEvent, MessageEventResult
from astrbot.core import logger
from astrbot.core.agent.runners.deerflow.constants import (
    DEERFLOW_AGENT_RUNNER_PROVIDER_ID_KEY,
    DEERFLOW_PROVIDER_TYPE,
    DEERFLOW_THREAD_ID_KEY,
)
from astrbot.core.agent.runners.deerflow.deerflow_api_client import DeerFlowAPIClient
from astrbot.core.db.po import ProviderStat
from astrbot.core.utils.active_event_registry import active_event_registry

from .utils.rst_scene import RstScene

THIRD_PARTY_AGENT_RUNNER_KEY = {
    "dify": "dify_conversation_id",
    "coze": "coze_conversation_id",
    "dashscope": "dashscope_conversation_id",
    DEERFLOW_PROVIDER_TYPE: DEERFLOW_THREAD_ID_KEY,
}
THIRD_PARTY_AGENT_RUNNER_STR = ", ".join(THIRD_PARTY_AGENT_RUNNER_KEY.keys())


async def _cleanup_deerflow_thread_if_present(
    context: star.Context,
    umo: str,
) -> None:
    try:
        thread_id = await sp.get_async(
            scope="umo",
            scope_id=umo,
            key=DEERFLOW_THREAD_ID_KEY,
            default="",
        )
        if not thread_id:
            return

        cfg = context.get_config(umo=umo)
        provider_id = cfg["provider_settings"].get(
            DEERFLOW_AGENT_RUNNER_PROVIDER_ID_KEY,
            "",
        )
        if not provider_id:
            return

        merged_provider_config = context.provider_manager.get_provider_config_by_id(
            provider_id,
            merged=True,
        )
        if not merged_provider_config:
            logger.warning(
                "Failed to resolve DeerFlow provider config for remote thread cleanup: provider_id=%s",
                provider_id,
            )
            return

        client = DeerFlowAPIClient(
            api_base=merged_provider_config.get(
                "deerflow_api_base",
                "http://127.0.0.1:2026",
            ),
            api_key=merged_provider_config.get("deerflow_api_key", ""),
            auth_header=merged_provider_config.get("deerflow_auth_header", ""),
            proxy=merged_provider_config.get("proxy", ""),
        )
        try:
            await client.delete_thread(thread_id)
        finally:
            try:
                await client.close()
            except Exception as e:
                logger.warning(
                    "Failed to close DeerFlow API client after thread cleanup: %s",
                    e,
                )
    except Exception as e:
        logger.warning(
            "Failed to clean up DeerFlow thread for session %s: %s",
            umo,
            e,
        )


async def _clear_third_party_agent_runner_state(
    context: star.Context,
    umo: str,
    agent_runner_type: str,
) -> None:
    session_key = THIRD_PARTY_AGENT_RUNNER_KEY.get(agent_runner_type)
    if not session_key:
        return

    if agent_runner_type == DEERFLOW_PROVIDER_TYPE:
        await _cleanup_deerflow_thread_if_present(context, umo)

    await sp.remove_async(
        scope="umo",
        scope_id=umo,
        key=session_key,
    )


class ConversationCommands:
    def __init__(self, context: star.Context) -> None:
        self.context = context

    async def _get_current_persona_id(self, session_id):
        curr = await self.context.conversation_manager.get_curr_conversation_id(
            session_id,
        )
        if not curr:
            return None
        conv = await self.context.conversation_manager.get_conversation(
            session_id,
            curr,
        )
        if not conv:
            return None
        return conv.persona_id

    async def reset(self, message: AstrMessageEvent) -> None:
        """重置 LLM 会话"""
        umo = message.unified_msg_origin
        cfg = self.context.get_config(umo=message.unified_msg_origin)
        is_unique_session = cfg["platform_settings"]["unique_session"]
        is_group = bool(message.get_group_id())

        scene = RstScene.get_scene(is_group, is_unique_session)

        alter_cmd_cfg = await sp.get_async("global", "global", "alter_cmd", {})
        plugin_config = alter_cmd_cfg.get("astrbot", {})
        reset_cfg = plugin_config.get("reset", {})

        required_perm = reset_cfg.get(
            scene.key,
            "admin" if is_group and not is_unique_session else "member",
        )

        if required_perm == "admin" and message.role != "admin":
            message.set_result(
                MessageEventResult().message(
                    f"Reset command requires admin permission in {scene.name} scenario, "
                    f"you (ID {message.get_sender_id()}) are not admin, cannot perform this action.",
                ),
            )
            return

        agent_runner_type = cfg["provider_settings"]["agent_runner_type"]
        if agent_runner_type in THIRD_PARTY_AGENT_RUNNER_KEY:
            active_event_registry.stop_all(umo, exclude=message)
            await _clear_third_party_agent_runner_state(
                self.context,
                umo,
                agent_runner_type,
            )
            message.set_result(
                MessageEventResult().message("✅ Conversation reset successfully.")
            )
            return

        if not self.context.get_using_provider(umo):
            message.set_result(
                MessageEventResult().message(
                    "😕 Cannot find any LLM provider. Configure one first."
                ),
            )
            return

        cid = await self.context.conversation_manager.get_curr_conversation_id(umo)

        if not cid:
            message.set_result(
                MessageEventResult().message(
                    "😕 You are not in a conversation. Use /new to create one.",
                ),
            )
            return

        active_event_registry.stop_all(umo, exclude=message)

        await self.context.conversation_manager.update_conversation(
            umo,
            cid,
            [],
        )

        ret = "✅ Conversation reset successfully."

        message.set_extra("_clean_ltm_session", True)

        message.set_result(MessageEventResult().message(ret))

    async def stop(self, message: AstrMessageEvent) -> None:
        """停止当前会话正在运行的 Agent"""
        cfg = self.context.get_config(umo=message.unified_msg_origin)
        agent_runner_type = cfg["provider_settings"]["agent_runner_type"]
        umo = message.unified_msg_origin

        if agent_runner_type in THIRD_PARTY_AGENT_RUNNER_KEY:
            stopped_count = active_event_registry.stop_all(umo, exclude=message)
        else:
            stopped_count = active_event_registry.request_agent_stop_all(
                umo,
                exclude=message,
            )

        if stopped_count > 0:
            message.set_result(
                MessageEventResult().message(
                    f"✅ Requested to stop {stopped_count} running tasks."
                )
            )
            return

        message.set_result(
            MessageEventResult().message("✅ No running tasks in the current session.")
        )

    async def new_conv(self, message: AstrMessageEvent) -> None:
        """创建新对话"""
        cfg = self.context.get_config(umo=message.unified_msg_origin)
        agent_runner_type = cfg["provider_settings"]["agent_runner_type"]
        if agent_runner_type in THIRD_PARTY_AGENT_RUNNER_KEY:
            active_event_registry.stop_all(message.unified_msg_origin, exclude=message)
            await _clear_third_party_agent_runner_state(
                self.context,
                message.unified_msg_origin,
                agent_runner_type,
            )
            message.set_result(
                MessageEventResult().message("✅ New conversation created.")
            )
            return

        active_event_registry.stop_all(message.unified_msg_origin, exclude=message)
        cpersona = await self._get_current_persona_id(message.unified_msg_origin)
        cid = await self.context.conversation_manager.new_conversation(
            message.unified_msg_origin,
            message.get_platform_id(),
            persona_id=cpersona,
        )

        message.set_extra("_clean_ltm_session", True)

        message.set_result(
            MessageEventResult().message(
                f"✅ Switched to new conversation: {cid[:4]}。"
            ),
        )

    async def stats(self, message: AstrMessageEvent) -> None:
        """Show token usage statistics for the current conversation."""
        umo = message.unified_msg_origin
        cid = await self.context.conversation_manager.get_curr_conversation_id(umo)

        if not cid:
            message.set_result(
                MessageEventResult().message(
                    "❌ You are not in a conversation. Use /new to create one."
                ),
            )
            return

        db = self.context.get_db()
        async with db.get_db() as session:
            result = await session.execute(
                select(
                    func.count(case((col(ProviderStat.id).is_not(None), 1))).label(
                        "record_count",
                    ),
                    func.coalesce(func.sum(ProviderStat.token_input_other), 0).label(
                        "total_input_other",
                    ),
                    func.coalesce(func.sum(ProviderStat.token_input_cached), 0).label(
                        "total_input_cached",
                    ),
                    func.coalesce(func.sum(ProviderStat.token_output), 0).label(
                        "total_output",
                    ),
                ).where(
                    col(ProviderStat.agent_type) == "internal",
                    col(ProviderStat.conversation_id) == cid,
                )
            )
            stats = result.one()

        provider_using = self.context.get_using_provider(umo=umo)
        provider_line = ""
        ctx_line = ""
        if provider_using:
            model_name = provider_using.get_model() or "(unknown)"
            provider_id = provider_using.meta().id
            provider_line = f"Model:          {provider_id} / {model_name}\n"

            max_ctx = self._resolve_max_context_tokens(provider_using, umo)
            current_tokens = await self._resolve_current_tokens(umo, cid)
            ctx_line = self._format_context_line(current_tokens, max_ctx)

        if stats.record_count == 0:
            header = f"📊 Conversation status (ID: {cid[:8]}...)\n"
            ret = header + provider_line + ctx_line
            ret += "📊 No token usage records for this conversation yet.\n"
            message.set_result(MessageEventResult().message(ret))
            return

        total_input_other = stats.total_input_other
        total_input_cached = stats.total_input_cached
        total_output = stats.total_output
        total_tokens = total_input_other + total_input_cached + total_output

        ret = (
            f"📊 Conversation status (ID: {cid[:8]}...)\n"
            f"{provider_line}"
            f"{ctx_line}"
            f"Total:          {total_tokens:,}\n"
            f"Input (cached): {total_input_cached:,}\n"
            f"Input (other):  {total_input_other:,}\n"
            f"Output:         {total_output:,}\n"
        )

        message.set_result(MessageEventResult().message(ret))

    def _resolve_max_context_tokens(self, provider, umo: str) -> int:
        """Resolve effective max_context_tokens for display.

        Mirrors the boot-time fill-in logic in astr_main_agent: explicit config
        wins; otherwise look up LLM_METADATAS by model name; otherwise fall back
        to provider_settings.fallback_max_context_tokens (default 128000)."""
        from astrbot.core.utils.llm_metadata import LLM_METADATAS

        configured = provider.provider_config.get("max_context_tokens", 0)
        if configured and configured > 0:
            return int(configured)

        model = provider.get_model()
        if model and (info := LLM_METADATAS.get(model)):
            limit = info.get("limit", {}).get("context")
            if limit:
                return int(limit)

        cfg = self.context.get_config(umo=umo).get("provider_settings", {})
        return int(cfg.get("fallback_max_context_tokens", 128000))

    async def _resolve_current_tokens(self, umo: str, cid: str) -> int:
        """Fetch the most recent LLM call's total tokens for this conversation.

        ConversationV2.token_usage is updated after each turn with the latest
        prompt+completion total — this is the closest stand-in for "how full is
        the context right now"."""
        conv = await self.context.conversation_manager.get_conversation(umo, cid)
        if not conv:
            return 0
        return int(getattr(conv, "token_usage", 0) or 0)

    def _format_context_line(self, current: int, max_ctx: int) -> str:
        """Format the context-window / compaction-threshold line."""
        if max_ctx <= 0:
            return "Context:        (max_context_tokens unknown)\n"

        # Built-in compressors default to 82% — note that custom compressors may
        # override this; the line is informational, not authoritative.
        threshold_ratio = 0.82
        threshold = int(max_ctx * threshold_ratio)
        if current <= 0:
            return (
                f"Context:        ?/{max_ctx:,} (compact at ~{threshold:,})\n"
            )

        pct = (current / max_ctx) * 100
        remaining = threshold - current
        if remaining > 0:
            tail = f"{remaining:,} until compact (@{threshold_ratio:.0%})"
        else:
            tail = f"over compact threshold by {-remaining:,}"
        return (
            f"Context:        {current:,}/{max_ctx:,} ({pct:.1f}%) — {tail}\n"
        )
