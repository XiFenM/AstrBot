import asyncio
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import telegramify_markdown
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReactionTypeCustomEmoji, ReactionTypeEmoji
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import ExtBot

from astrbot import logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import (
    At,
    File,
    Image,
    Plain,
    Record,
    Reply,
    Video,
)
from astrbot.api.platform import AstrBotMessage, MessageType, PlatformMetadata
from astrbot.core.utils.metrics import Metric


@dataclass
class TelegramInlineKeyboard:
    """Telegram 内联键盘组件（仅用于 Telegram 适配器）。

    将此对象加入 MessageChain，Telegram 适配器会在发送文本消息时附带内联按钮。

    Attributes:
        buttons: 按钮行列表，每行是 (显示文本, callback_data) 元组的列表。
    """

    buttons: list[list[tuple[str, str]]]
    type: str = "TelegramInlineKeyboard"


class TelegramHTMLText:
    """Telegram HTML 格式文本组件（仅用于 Telegram 适配器）。

    将此对象加入 MessageChain，Telegram 适配器会直接以 HTML parse_mode 发送，
    跳过 telegramify_markdown 转换。适用于 expandable blockquote 等原生 HTML 特性。
    """

    def __init__(self, html: str):
        self.html = html
        self.type = "TelegramHTMLText"


def _is_gif(path: str) -> bool:
    if path.lower().endswith(".gif"):
        return True
    try:
        with open(path, "rb") as f:
            return f.read(6) in (b"GIF87a", b"GIF89a")
    except OSError:
        return False

class _StreamingCtx:
    """流式输出期间的共享可变状态。

    ``_send_streaming_draft`` / ``_send_streaming_edit`` 运行期间创建并挂到
    ``TelegramPlatformEvent._streaming_ctx`` 上；``send()`` 检测到该对象存在且
    ``delta`` 非空时，会先 flush 已累积的文本为真实消息，再发送新消息。
    这样即使 ``show_tool_use=False``，工具审批等通过 ``event.send()`` 插入的独立
    消息也不会与流式缓冲冲突。
    """

    __slots__ = (
        "mode", "delta", "last_sent_text", "draft_id",
        "payload", "user_name", "message_thread_id",
        "text_changed", "message_id", "event_ref",
    )

    def __init__(
        self,
        mode: str,
        payload: dict[str, Any],
        user_name: str,
        message_thread_id: str | None,
        event_ref: "TelegramPlatformEvent",
    ) -> None:
        self.mode = mode  # "draft" | "edit"
        self.delta = ""
        self.last_sent_text = ""
        self.draft_id: int = 0  # draft 模式专用
        self.payload = payload
        self.user_name = user_name
        self.message_thread_id = message_thread_id
        self.message_id: int | None = None  # edit 模式专用
        self.text_changed: asyncio.Event = asyncio.Event()
        self.event_ref = event_ref

    async def flush(self) -> None:
        """将已累积的 delta 刷为真实消息，重置缓冲区。"""
        if not self.delta:
            return
        text = self.delta

        if self.mode == "draft":
            # 清空 draft 显示
            await self.event_ref._send_message_draft(
                self.user_name, self.draft_id, "\u23f3", self.message_thread_id,
            )
            await self.event_ref._send_final_segment(text, self.payload)
            self.draft_id = TelegramPlatformEvent._allocate_draft_id()
        else:
            # edit 模式：如果有 message_id，最终编辑一次
            if self.message_id:
                try:
                    await self.event_ref.client.edit_message_text(
                        text=text,
                        chat_id=self.payload["chat_id"],
                        message_id=self.message_id,
                    )
                except Exception as e:
                    logger.warning(f"flush edit_message_text failed: {e!s}")
            self.message_id = None

        self.delta = ""
        self.last_sent_text = ""


class TelegramPlatformEvent(AstrMessageEvent):
    # Telegram 的最大消息长度限制
    MAX_MESSAGE_LENGTH = 4096

    SPLIT_PATTERNS = {
        "paragraph": re.compile(r"\n\n"),
        "line": re.compile(r"\n"),
        "sentence": re.compile(r"[.!?。！？]"),
        "word": re.compile(r"\s"),
    }

    # sendMessageDraft 的 draft_id 类级递增计数器
    _TELEGRAM_DRAFT_ID_MAX = 2_147_483_647
    _next_draft_id: int = 0

    @classmethod
    def _allocate_draft_id(cls) -> int:
        """分配一个递增的 draft_id，溢出时归 1。"""
        cls._next_draft_id = (
            1
            if cls._next_draft_id >= cls._TELEGRAM_DRAFT_ID_MAX
            else cls._next_draft_id + 1
        )
        return cls._next_draft_id

    # 消息类型到 chat action 的映射，用于优先级判断
    ACTION_BY_TYPE: dict[type, str] = {
        Record: ChatAction.UPLOAD_VOICE,
        Video: ChatAction.UPLOAD_VIDEO,
        File: ChatAction.UPLOAD_DOCUMENT,
        Image: ChatAction.UPLOAD_PHOTO,
        Plain: ChatAction.TYPING,
    }

    def __init__(
        self,
        message_str: str,
        message_obj: AstrBotMessage,
        platform_meta: PlatformMetadata,
        session_id: str,
        client: ExtBot,
    ) -> None:
        super().__init__(message_str, message_obj, platform_meta, session_id)
        self.client = client

        # ── 流式输出状态（供 send() 在流式期间自动 flush） ──
        # _send_streaming_draft / _send_streaming_edit 运行期间设置，
        # send() 检测到有累积内容时会先 flush 再发送新消息。
        self._streaming_ctx: _StreamingCtx | None = None

    @classmethod
    def _split_message(cls, text: str) -> list[str]:
        if len(text) <= cls.MAX_MESSAGE_LENGTH:
            return [text]

        chunks = []
        while text:
            if len(text) <= cls.MAX_MESSAGE_LENGTH:
                chunks.append(text)
                break

            split_point = cls.MAX_MESSAGE_LENGTH
            segment = text[: cls.MAX_MESSAGE_LENGTH]

            for _, pattern in cls.SPLIT_PATTERNS.items():
                if matches := list(pattern.finditer(segment)):
                    last_match = matches[-1]
                    split_point = last_match.end()
                    break

            chunks.append(text[:split_point])
            text = text[split_point:].lstrip()

        return chunks

    @classmethod
    async def _send_chat_action(
        cls,
        client: ExtBot,
        chat_id: str,
        action: ChatAction | str,
        message_thread_id: str | None = None,
    ) -> None:
        """发送聊天状态动作"""
        try:
            payload: dict[str, Any] = {"chat_id": chat_id, "action": action}
            if message_thread_id:
                payload["message_thread_id"] = message_thread_id
            await client.send_chat_action(**payload)
        except Exception as e:
            logger.warning(f"[Telegram] 发送 chat action 失败: {e}")

    @classmethod
    def _get_chat_action_for_chain(cls, chain: list[Any]) -> ChatAction | str:
        """根据消息链中的组件类型确定合适的 chat action（按优先级）"""
        for seg_type, action in cls.ACTION_BY_TYPE.items():
            if any(isinstance(seg, seg_type) for seg in chain):
                return action
        return ChatAction.TYPING

    @classmethod
    async def _send_media_with_action(
        cls,
        client: ExtBot,
        upload_action: ChatAction | str,
        send_coro,
        *,
        user_name: str,
        message_thread_id: str | None = None,
        **payload: Any,
    ) -> None:
        """发送媒体时显示 upload action，发送完成后恢复 typing"""
        effective_thread_id = message_thread_id or cast(
            str | None, payload.get("message_thread_id")
        )
        await cls._send_chat_action(
            client, user_name, upload_action, effective_thread_id
        )
        send_payload = dict(payload)
        if effective_thread_id and "message_thread_id" not in send_payload:
            send_payload["message_thread_id"] = effective_thread_id
        await send_coro(**send_payload)
        await cls._send_chat_action(
            client, user_name, ChatAction.TYPING, effective_thread_id
        )

    @classmethod
    async def _send_voice_with_fallback(
        cls,
        client: ExtBot,
        path: str,
        payload: dict[str, Any],
        *,
        caption: str | None = None,
        user_name: str = "",
        message_thread_id: str | None = None,
        use_media_action: bool = False,
    ) -> None:
        """Send a voice message, falling back to a document if the user's
        privacy settings forbid voice messages (``BadRequest`` with
        ``Voice_messages_forbidden``).

        When *use_media_action* is ``True`` the helper wraps the send calls
        with ``_send_media_with_action`` (used by the streaming path).
        """
        try:
            if use_media_action:
                media_payload = dict(payload)
                if message_thread_id and "message_thread_id" not in media_payload:
                    media_payload["message_thread_id"] = message_thread_id
                await cls._send_media_with_action(
                    client,
                    ChatAction.UPLOAD_VOICE,
                    client.send_voice,
                    user_name=user_name,
                    voice=path,
                    **cast(Any, media_payload),
                )
            else:
                await client.send_voice(voice=path, **cast(Any, payload))
        except BadRequest as e:
            # python-telegram-bot raises BadRequest for Voice_messages_forbidden;
            # distinguish the voice-privacy case via the API error message.
            if "Voice_messages_forbidden" not in e.message:
                raise
            logger.warning(
                "User privacy settings prevent receiving voice messages, falling back to sending an audio file. "
                "To enable voice messages, go to Telegram Settings → Privacy and Security → Voice Messages → set to 'Everyone'."
            )
            if use_media_action:
                media_payload = dict(payload)
                if message_thread_id and "message_thread_id" not in media_payload:
                    media_payload["message_thread_id"] = message_thread_id
                await cls._send_media_with_action(
                    client,
                    ChatAction.UPLOAD_DOCUMENT,
                    client.send_document,
                    user_name=user_name,
                    document=path,
                    caption=caption,
                    **cast(Any, media_payload),
                )
            else:
                await client.send_document(
                    document=path,
                    caption=caption,
                    **cast(Any, payload),
                )

    async def _ensure_typing(
        self,
        user_name: str,
        message_thread_id: str | None = None,
    ) -> None:
        """确保显示 typing 状态"""
        await self._send_chat_action(
            self.client, user_name, ChatAction.TYPING, message_thread_id
        )

    async def send_typing(self) -> None:
        message_thread_id = None
        if self.get_message_type() == MessageType.GROUP_MESSAGE:
            user_name = self.message_obj.group_id
        else:
            user_name = self.get_sender_id()

        if "#" in user_name:
            user_name, message_thread_id = user_name.split("#")

        await self._ensure_typing(user_name, message_thread_id)

    @classmethod
    async def send_with_client(
        cls,
        client: ExtBot,
        message: MessageChain,
        user_name: str,
    ) -> None:
        image_path = None

        has_reply = False
        reply_message_id = None
        at_user_id = None
        for i in message.chain:
            if isinstance(i, Reply):
                has_reply = True
                reply_message_id = i.id
            if isinstance(i, At):
                at_user_id = i.name

        at_flag = False
        message_thread_id = None
        if "#" in user_name:
            # it's a supergroup chat with message_thread_id
            user_name, message_thread_id = user_name.split("#")

        # 收集内联键盘（如有），并记录最后一个文本组件的索引，键盘将附加在那里
        _reply_markup: InlineKeyboardMarkup | None = None
        _last_text_idx: int | None = None
        for _idx, _item in enumerate(message.chain):
            if isinstance(_item, TelegramInlineKeyboard):
                _reply_markup = InlineKeyboardMarkup([
                    [InlineKeyboardButton(text=label, callback_data=data) for label, data in row]
                    for row in _item.buttons
                ])
            if isinstance(_item, (Plain, TelegramHTMLText)):
                _last_text_idx = _idx

        # 根据消息链确定合适的 chat action 并发送
        action = cls._get_chat_action_for_chain(message.chain)
        await cls._send_chat_action(client, user_name, action, message_thread_id)

        for chain_idx, i in enumerate(message.chain):
            payload = {
                "chat_id": user_name,
            }
            if has_reply:
                payload["reply_to_message_id"] = str(reply_message_id)
            if message_thread_id:
                payload["message_thread_id"] = message_thread_id

            if isinstance(i, Plain):
                if at_user_id and not at_flag:
                    i.text = f"@{at_user_id} {i.text}"
                    at_flag = True
                chunks = cls._split_message(i.text)
                for chunk_idx, chunk in enumerate(chunks):
                    # 仅在最后一个文本组件的最后一个分块上附加键盘
                    is_last_chunk = (
                        chain_idx == _last_text_idx and chunk_idx == len(chunks) - 1
                    )
                    markup = _reply_markup if is_last_chunk else None
                    try:
                        md_text = telegramify_markdown.markdownify(
                            chunk,
                        )
                        await client.send_message(
                            text=md_text,
                            parse_mode="MarkdownV2",
                            reply_markup=markup,
                            **cast(Any, payload),
                        )
                    except Exception as e:
                        logger.warning(
                            f"MarkdownV2 send failed: {e}. Using plain text instead.",
                        )
                        await client.send_message(
                            text=chunk,
                            reply_markup=markup,
                            **cast(Any, payload),
                        )
            elif isinstance(i, TelegramHTMLText):
                markup = _reply_markup if chain_idx == _last_text_idx else None
                await client.send_message(
                    text=i.html,
                    parse_mode="HTML",
                    reply_markup=markup,
                    **cast(Any, payload),
                )
            elif isinstance(i, Image):
                image_path = await i.convert_to_file_path()
                if _is_gif(image_path):
                    send_coro = client.send_animation
                    media_kwarg = {"animation": image_path}
                else:
                    send_coro = client.send_photo
                    media_kwarg = {"photo": image_path}
                await send_coro(**media_kwarg, **cast(Any, payload))
            elif isinstance(i, File):
                path = await i.get_file()
                name = i.name or os.path.basename(path)
                await client.send_document(
                    document=path, filename=name, **cast(Any, payload)
                )
            elif isinstance(i, Record):
                path = await i.convert_to_file_path()
                await cls._send_voice_with_fallback(
                    client,
                    path,
                    payload,
                    caption=i.text or None,
                    use_media_action=False,
                )
            elif isinstance(i, Video):
                path = await i.convert_to_file_path()
                await client.send_video(
                    video=path,
                    caption=getattr(i, "text", None) or None,
                    **cast(Any, payload),
                )

    async def send(self, message: MessageChain) -> None:
        # 流式输出期间有独立消息插入 → 先 flush 已累积的流式文本
        if self._streaming_ctx and self._streaming_ctx.delta:
            await self._streaming_ctx.flush()

        if self.get_message_type() == MessageType.GROUP_MESSAGE:
            await self.send_with_client(self.client, message, self.message_obj.group_id)
        else:
            await self.send_with_client(self.client, message, self.get_sender_id())
        await super().send(message)

    async def react(self, emoji: str | None, big: bool = False) -> None:
        """给原消息添加 Telegram 反应：
        - 普通 emoji：传入 '👍'、'😂' 等
        - 自定义表情：传入其 custom_emoji_id（纯数字字符串）
        - 取消本机器人的反应：传入 None 或空字符串
        """
        try:
            # 解析 chat_id（去掉超级群的 "#<thread_id>" 片段）
            if self.get_message_type() == MessageType.GROUP_MESSAGE:
                chat_id = (self.message_obj.group_id or "").split("#")[0]
            else:
                chat_id = self.get_sender_id()

            message_id = int(self.message_obj.message_id)

            # 组装 reaction 参数（必须是 ReactionType 的列表）
            if not emoji:  # 清空本 bot 的反应
                reaction_param = []  # 空列表表示移除本 bot 的反应
            elif emoji.isdigit():  # 自定义表情：传 custom_emoji_id
                reaction_param = [ReactionTypeCustomEmoji(emoji)]
            else:  # 普通 emoji
                reaction_param = [ReactionTypeEmoji(emoji)]

            await self.client.set_message_reaction(
                chat_id=chat_id,
                message_id=message_id,
                reaction=reaction_param,  # 注意是列表
                is_big=big,  # 可选：大动画
            )
        except Exception as e:
            logger.error(f"[Telegram] 添加反应失败: {e}")

    async def _send_message_draft(
        self,
        chat_id: str,
        draft_id: int,
        text: str,
        message_thread_id: str | None = None,
        parse_mode: str | None = None,
    ) -> None:
        """通过 Bot.send_message_draft 发送草稿消息（流式推送部分消息）。

        该 API 仅支持私聊。

        Args:
            chat_id: 目标私聊的 chat_id
            draft_id: 草稿唯一标识，非零整数；相同 draft_id 的变更会以动画展示
            text: 消息文本，1-4096 字符
            message_thread_id: 可选，目标消息线程 ID
            parse_mode: 可选，消息文本的解析模式
        """
        kwargs: dict[str, Any] = {}
        if message_thread_id:
            kwargs["message_thread_id"] = int(message_thread_id)
        if parse_mode:
            kwargs["parse_mode"] = parse_mode

        try:
            logger.debug(
                f"[Telegram] sendMessageDraft: chat_id={chat_id}, draft_id={draft_id}, text_len={len(text)}"
            )
            await self.client.send_message_draft(
                chat_id=int(chat_id),
                draft_id=draft_id,
                text=text,
                **kwargs,
            )
        except Exception as e:
            logger.warning(f"[Telegram] sendMessageDraft 失败: {e!s}")

    async def _process_chain_items(
        self,
        chain: MessageChain,
        payload: dict[str, Any],
        user_name: str,
        message_thread_id: str | None,
        on_text: Callable[[str], None],
    ) -> None:
        """处理 MessageChain 中的各类组件，文本通过 on_text 回调追加，媒体直接发送。"""
        for i in chain.chain:
            if isinstance(i, Plain):
                on_text(i.text)
            elif isinstance(i, Image):
                image_path = await i.convert_to_file_path()
                if _is_gif(image_path):
                    action = ChatAction.UPLOAD_VIDEO
                    send_coro = self.client.send_animation
                    media_kwarg = {"animation": image_path}
                else:
                    action = ChatAction.UPLOAD_PHOTO
                    send_coro = self.client.send_photo
                    media_kwarg = {"photo": image_path}
                await self._send_media_with_action(
                    self.client,
                    action,
                    send_coro,
                    user_name=user_name,
                    **media_kwarg,
                    **cast(Any, payload),
                )
            elif isinstance(i, File):
                path = await i.get_file()
                name = i.name or os.path.basename(path)
                await self._send_media_with_action(
                    self.client,
                    ChatAction.UPLOAD_DOCUMENT,
                    self.client.send_document,
                    user_name=user_name,
                    document=path,
                    filename=name,
                    **cast(Any, payload),
                )
            elif isinstance(i, Record):
                path = await i.convert_to_file_path()
                await self._send_voice_with_fallback(
                    self.client,
                    path,
                    payload,
                    caption=i.text or None,
                    user_name=user_name,
                    message_thread_id=message_thread_id,
                    use_media_action=True,
                )
            elif isinstance(i, Video):
                path = await i.convert_to_file_path()
                await self._send_media_with_action(
                    self.client,
                    ChatAction.UPLOAD_VIDEO,
                    self.client.send_video,
                    user_name=user_name,
                    video=path,
                    **cast(Any, payload),
                )
            else:
                logger.warning(f"不支持的消息类型: {type(i)}")

    async def _send_final_segment(self, delta: str, payload: dict[str, Any]) -> None:
        """将累积文本作为 MarkdownV2 真实消息发送，失败时回退到纯文本。"""
        try:
            markdown_text = telegramify_markdown.markdownify(
                delta,
            )
            await self.client.send_message(
                text=markdown_text,
                parse_mode="MarkdownV2",
                **cast(Any, payload),
            )
        except Exception as e:
            logger.warning(f"Markdown转换失败，使用普通文本: {e!s}")
            await self.client.send_message(text=delta, **cast(Any, payload))

    async def send_streaming(self, generator, use_fallback: bool = False):
        msg_type = self.get_message_type()
        message_thread_id = None

        if msg_type == MessageType.GROUP_MESSAGE:
            user_name = self.message_obj.group_id
        else:
            user_name = self.get_sender_id()

        if "#" in user_name:
            # it's a supergroup chat with message_thread_id
            user_name, message_thread_id = user_name.split("#")
        payload = {
            "chat_id": user_name,
        }
        if message_thread_id:
            payload["message_thread_id"] = message_thread_id

        # sendMessageDraft 仅支持私聊（显式检查 FRIEND_MESSAGE）
        is_private = self.get_message_type() == MessageType.FRIEND_MESSAGE

        if is_private:
            logger.info("[Telegram] 流式输出: 使用 sendMessageDraft (私聊)")
            await self._send_streaming_draft(
                user_name, message_thread_id, payload, generator
            )
        else:
            logger.info("[Telegram] 流式输出: 使用 edit_message_text fallback (群聊)")
            await self._send_streaming_edit(
                user_name, message_thread_id, payload, generator
            )

        # 内联父类 send_streaming 的副作用（避免传入已消费的 generator）
        asyncio.create_task(
            Metric.upload(msg_event_tick=1, adapter_name=self.platform_meta.name),
        )
        self._has_send_oper = True

    async def _send_streaming_draft(
        self,
        user_name: str,
        message_thread_id: str | None,
        payload: dict[str, Any],
        generator,
    ) -> None:
        """使用 sendMessageDraft API 进行流式推送（私聊专用）。

        流式过程中使用 sendMessageDraft 推送草稿动画，
        流式结束后发送一条真实消息保留最终内容（draft 是临时的，会消失）。
        使用信号驱动的发送循环：每次有新 token 到达时唤醒发送，
        发送频率由网络 RTT 自然限制（最多一个请求 in-flight）。

        通过 ``self._streaming_ctx`` 与 ``send()`` 共享缓冲区状态：
        若工具执行中通过 ``event.send()`` 插入独立消息（如审批通知），
        ``send()`` 会自动先 flush 已累积的文本为真实消息，保证消息顺序。
        """
        ctx = _StreamingCtx("draft", payload, user_name, message_thread_id, self)
        ctx.draft_id = self._allocate_draft_id()
        self._streaming_ctx = ctx

        done = False  # 信号：生成器已结束

        async def _draft_sender_loop() -> None:
            """信号驱动的草稿发送循环，有新内容就发，RTT 自然限流。"""
            while not done:
                await ctx.text_changed.wait()
                ctx.text_changed.clear()
                # 发送最新的缓冲区内容（MarkdownV2 渲染，与真实消息一致）
                if ctx.delta and ctx.delta != ctx.last_sent_text:
                    draft_text = ctx.delta[: self.MAX_MESSAGE_LENGTH]
                    if draft_text != ctx.last_sent_text:
                        try:
                            md = telegramify_markdown.markdownify(
                                draft_text,
                            )
                            await self._send_message_draft(
                                user_name,
                                ctx.draft_id,
                                md,
                                message_thread_id,
                                parse_mode="MarkdownV2",
                            )
                            ctx.last_sent_text = draft_text
                        except Exception:
                            # markdownify 对未闭合语法可能失败，回退纯文本
                            try:
                                await self._send_message_draft(
                                    user_name,
                                    ctx.draft_id,
                                    draft_text,
                                    message_thread_id,
                                )
                                ctx.last_sent_text = draft_text
                            except Exception as e2:
                                logger.debug(
                                    f"[Telegram] sendMessageDraft failed (ignored): {e2!s}"
                                )

        sender_task = asyncio.create_task(_draft_sender_loop())

        def _append_text(t: str) -> None:
            ctx.delta += t
            ctx.text_changed.set()  # 唤醒发送循环

        try:
            async for chain in generator:
                if not isinstance(chain, MessageChain):
                    continue

                if chain.type == "break":
                    # 分割符：发送真实消息保留内容，重置缓冲区
                    await ctx.flush()
                    continue

                await self._process_chain_items(
                    chain, payload, user_name, message_thread_id, _append_text
                )
        finally:
            done = True
            ctx.text_changed.set()  # 唤醒循环使其退出
            await sender_task
            self._streaming_ctx = None

        # 流式结束：flush 剩余内容
        if ctx.delta:
            await self._send_message_draft(
                user_name,
                ctx.draft_id,
                "\u23f3",
                message_thread_id,
            )
            await self._send_final_segment(ctx.delta, payload)

    async def _send_streaming_edit(
        self,
        user_name: str,
        message_thread_id: str | None,
        payload: dict[str, Any],
        generator,
    ) -> None:
        """使用 send_message + edit_message_text 进行流式推送（群聊 fallback）。

        通过 ``self._streaming_ctx`` 与 ``send()`` 共享缓冲区状态，
        实现流式期间独立消息插入时的自动 flush。
        """
        ctx = _StreamingCtx("edit", payload, user_name, message_thread_id, self)
        self._streaming_ctx = ctx

        current_content = ""
        last_edit_time = 0  # 上次编辑消息的时间
        throttle_interval = 0.6  # 编辑消息的间隔时间 (秒)
        last_chat_action_time = 0  # 上次发送 chat action 的时间
        chat_action_interval = 0.5  # chat action 的节流间隔 (秒)

        # 发送初始 typing 状态
        await self._ensure_typing(user_name, message_thread_id)
        last_chat_action_time = asyncio.get_running_loop().time()

        def _append_text(t: str) -> None:
            ctx.delta += t

        try:
            async for chain in generator:
                if not isinstance(chain, MessageChain):
                    continue

                if chain.type == "break":
                    # 分割符
                    await ctx.flush()
                    current_content = ""
                    continue

                await self._process_chain_items(
                    chain, payload, user_name, message_thread_id, _append_text
                )

                # flush 可能被 send() 触发过，此时 message_id 已重置
                # 编辑或发送消息
                if ctx.message_id and len(ctx.delta) <= self.MAX_MESSAGE_LENGTH:
                    current_time = asyncio.get_running_loop().time()
                    time_since_last_edit = current_time - last_edit_time

                    if time_since_last_edit >= throttle_interval:
                        current_time = asyncio.get_running_loop().time()
                        if current_time - last_chat_action_time >= chat_action_interval:
                            await self._ensure_typing(user_name, message_thread_id)
                            last_chat_action_time = current_time
                        try:
                            await self.client.edit_message_text(
                                text=ctx.delta,
                                chat_id=payload["chat_id"],
                                message_id=ctx.message_id,
                            )
                            current_content = ctx.delta
                        except Exception as e:
                            logger.warning(f"编辑消息失败(streaming): {e!s}")
                        last_edit_time = asyncio.get_running_loop().time()
                else:
                    current_time = asyncio.get_running_loop().time()
                    if current_time - last_chat_action_time >= chat_action_interval:
                        await self._ensure_typing(user_name, message_thread_id)
                        last_chat_action_time = current_time
                    try:
                        msg = await self.client.send_message(
                            text=ctx.delta, **cast(Any, payload)
                        )
                        current_content = ctx.delta
                    except Exception as e:
                        logger.warning(f"发送消息失败(streaming): {e!s}")
                    ctx.message_id = msg.message_id
                    last_edit_time = asyncio.get_running_loop().time()
        finally:
            self._streaming_ctx = None

        try:
            if ctx.delta and current_content != ctx.delta:
                try:
                    markdown_text = telegramify_markdown.markdownify(
                        ctx.delta,
                    )
                    await self.client.edit_message_text(
                        text=markdown_text,
                        chat_id=payload["chat_id"],
                        message_id=ctx.message_id,
                        parse_mode="MarkdownV2",
                    )
                except Exception as e:
                    logger.warning(f"Markdown转换失败，使用普通文本: {e!s}")
                    await self.client.edit_message_text(
                        text=ctx.delta,
                        chat_id=payload["chat_id"],
                        message_id=ctx.message_id,
                    )
        except Exception as e:
            logger.warning(f"编辑消息失败(streaming): {e!s}")
