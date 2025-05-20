#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Novel writing agent using MetaGPT and LangChain.

This script demonstrates how one might build a multi-step novel writing
agent that mirrors the logic of the provided DeepSeek example. It relies
on LangChain for LLM interactions and sketches out a MetaGPT style agent
structure. The actual MetaGPT and LangChain packages are not included in
this repository, so this file serves as a conceptual template.
"""

from typing import List, Dict

# LangChain imports (placeholders; packages are not installed in this repo)
try:
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage
except ImportError:  # pragma: no cover - library not installed
    ChatOpenAI = object  # type: ignore
    SystemMessage = HumanMessage = object  # type: ignore

# MetaGPT imports (placeholders; packages are not installed in this repo)
try:
    from metagpt.actions import Action
    from metagpt.agent import Agent
except ImportError:  # pragma: no cover - library not installed
    Action = Agent = object  # type: ignore


# ================================
# 1) LLM call wrapped with LangChain
# ================================

def call_deepseek_chat(
    messages: List[Dict[str, str]],
    model: str = "deepseek-chat",
    temperature: float = 0.7,
) -> str:
    """Call the DeepSeek API via LangChain's ChatOpenAI wrapper."""
    base_url = "https://api.deepseek.com/v1"
    api_key = "YOUR_DEEPSEEK_KEY"  # TODO: replace with actual key

    # Convert dict messages to LangChain schema objects if possible
    chat_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            chat_messages.append(SystemMessage(content=content))
        else:
            chat_messages.append(HumanMessage(content=content))

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_base=base_url,
        openai_api_key=api_key,
    )
    response = llm(chat_messages)
    return getattr(response, "content", "")


# ================================
# 2) Actions representing outline, writing, review, and summary
# ================================

class GenerateOutline(Action):
    """Action to create a novel outline."""

    async def run(
        self,
        title: str,
        theme: str,
        main_characters: Dict[str, str],
        plot_summary: str,
        chapter_count: int,
    ) -> str:
        system_prompt = (
            "你是一位专职于写作规划的AI，请根据用户提供的信息生成详细的小说大纲，"
            "包括章节的主要事件、转折等。"
        )
        user_prompt = (
            f"请根据以下信息生成小说大纲：\n"
            f"【标题】{title}\n"
            f"【主题】{theme}\n"
            f"【主要角色】{main_characters}\n"
            f"【剧情概述】{plot_summary}\n"
            f"【预计章节数】{chapter_count}\n"
            "请用分章节的方式列出主要内容，并保持条理清晰。"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return call_deepseek_chat(messages)


class WriteChapter(Action):
    """Action to write a single chapter."""

    async def run(
        self,
        outline: str,
        prev_summaries: str,
        chapter_title: str,
        chapter_index: int,
        target_word_count: int = 2000,
    ) -> str:
        system_prompt = (
            "你是一位优秀的小说作者，请根据用户提供的小说大纲和前文摘要，"
            "写出连贯、完整的本章节正文，文笔优美并注意与前文呼应。"
        )
        user_prompt = (
            f"【小说大纲】\n{outline}\n\n"
            f"【前文摘要】\n{prev_summaries}\n\n"
            f"请写第{chapter_index}章：《{chapter_title}》，目标字数约 {target_word_count}。"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return call_deepseek_chat(messages)


class ReviewChapter(Action):
    """Action to review chapter text."""

    async def run(self, chapter_text: str) -> str:
        system_prompt = (
            "你是一位资深编辑，请对输入的小说章节内容进行审阅，指出可改进之处，"
            "如情节节奏、人物塑造、文笔风格等，并给出建议。"
        )
        user_prompt = (
            f"这是章节内容：\n\n{chapter_text}\n\n"
            "请给出详细的修改意见和建议。"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return call_deepseek_chat(messages)


class SummarizeChapter(Action):
    """Action to summarize chapter text."""

    async def run(self, chapter_text: str) -> str:
        system_prompt = (
            "你是一位文字总结高手，请为下面的小说章节生成简短且信息密集的摘要，"
            "突出本章主要事件、角色发展、与后续剧情可能的衔接。"
        )
        user_prompt = (
            f"章节内容：\n\n{chapter_text}\n\n"
            "请在200字以内给出本章的精简摘要。"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return call_deepseek_chat(messages)


# ================================
# 3) Agent orchestrating the full workflow
# ================================

class NovelAgent(Agent):  # type: ignore
    """A simple MetaGPT style agent managing novel creation."""

    def __init__(self):
        super().__init__()
        self.outline_action = GenerateOutline()
        self.write_action = WriteChapter()
        self.review_action = ReviewChapter()
        self.summarize_action = SummarizeChapter()

    async def run(self) -> None:
        title = input("请输入小说标题：\n> ")
        theme = input("请输入小说主题或背景：\n> ")
        plot_summary = input("请输入故事的概述：\n> ")
        chapter_count_str = input("你想要多少章？(默认5)：\n> ")
        try:
            chapter_count = int(chapter_count_str.strip())
        except ValueError:
            chapter_count = 5

        main_characters = {
            "主角": "默认写一个大概的主角信息",
            "配角A": "可自行扩展",
        }

        print("\n[系统] 正在生成大纲，请稍候...\n")
        outline = await self.outline_action.run(
            title=title,
            theme=theme,
            main_characters=main_characters,
            plot_summary=plot_summary,
            chapter_count=chapter_count,
        )
        print("=== 小说大纲 ===")
        print(outline)

        all_chapters = []
        all_summaries = []

        for i in range(1, chapter_count + 1):
            chapter_title = f"{title} - 第{i}章"
            prev_summary = "(无)" if i == 1 else "\n".join(all_summaries)

            print(f"\n[系统] 开始写第{i}章：《{chapter_title}》...")
            chapter_text = await self.write_action.run(
                outline=outline,
                prev_summaries=prev_summary,
                chapter_title=chapter_title,
                chapter_index=i,
            )
            print(f"\n=== 第{i}章初稿 ===\n{chapter_text}")

            print("\n[系统] 正在审阅本章...")
            review_text = await self.review_action.run(chapter_text=chapter_text)
            print(f"=== 审阅意见 ===\n{review_text}\n")

            summary_text = await self.summarize_action.run(chapter_text=chapter_text)
            print(f"=== 本章摘要 ===\n{summary_text}\n")

            all_chapters.append(chapter_text)
            all_summaries.append(summary_text)

        print("\n\n=== 全部章节完成 ===\n")
        full_novel = "\n".join(all_chapters)
        print("=== 以下是完整小说内容 ===\n")
        print(full_novel)
        print("\n=== 写作结束，感谢使用！ ===")


if __name__ == "__main__":
    # The agent is defined as asynchronous in this example.
    import asyncio

    agent = NovelAgent()
    asyncio.run(agent.run())
