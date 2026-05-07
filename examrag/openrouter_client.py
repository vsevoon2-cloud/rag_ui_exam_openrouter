import base64
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _data_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


@dataclass
class OpenRouterUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


class OpenRouterClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()

    def _post(self, payload: dict[str, Any], timeout_s: int = 180, retries: int = 4) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                r = self.session.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=timeout_s)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(min(1.5 * (attempt + 1), 6.0))
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(min(1.5 * (attempt + 1), 6.0))
            except Exception as exc:
                last_error = exc
                break
        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenRouter request failed without error details")

    @staticmethod
    def _json_complete_heuristic(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        # Fast check: balanced braces/brackets outside quotes.
        stack = []
        in_string = False
        escape = False
        for ch in s:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            else:
                if ch == '"':
                    in_string = True
                    continue
                if ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if not stack:
                        return False
                    opener = stack.pop()
                    if (opener, ch) not in (("{", "}"), ("[", "]")):
                        return False
        return (not in_string) and (len(stack) == 0)

    @staticmethod
    def _extract_text(resp: dict[str, Any]) -> str:
        content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            parts = [str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text"]
            return "\n".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _extract_usage(resp: dict[str, Any]) -> OpenRouterUsage:
        usage = resp.get("usage") or {}
        try:
            return OpenRouterUsage(
                prompt_tokens=int(usage["prompt_tokens"]) if "prompt_tokens" in usage else None,
                completion_tokens=int(usage["completion_tokens"]) if "completion_tokens" in usage else None,
                total_tokens=int(usage["total_tokens"]) if "total_tokens" in usage else None,
                cost=float(usage["cost"]) if "cost" in usage else None,
            )
        except Exception:
            return OpenRouterUsage()

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        return text.strip()

    def _call_json_object_with_retry(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.1,
        retries: int = 2,
        timeout_s: int = 180,
    ) -> tuple[dict[str, Any], OpenRouterUsage, str]:
        """
        Calls OpenRouter with response_format json_object and retries once if the JSON is truncated/invalid.
        Returns (parsed_json_or_empty_dict, aggregated_usage, raw_text_last).
        """
        usage_total = OpenRouterUsage()
        last_raw = ""
        attempt_messages = list(messages)

        for attempt in range(retries + 1):
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
                "messages": attempt_messages,
            }
            resp = self._post(payload, timeout_s=timeout_s)
            raw = self._extract_text(resp)
            last_raw = raw
            usage = self._extract_usage(resp)
            usage_total.prompt_tokens = (usage_total.prompt_tokens or 0) + (usage.prompt_tokens or 0)
            usage_total.completion_tokens = (usage_total.completion_tokens or 0) + (usage.completion_tokens or 0)
            usage_total.total_tokens = (usage_total.total_tokens or 0) + (usage.total_tokens or 0)
            usage_total.cost = (usage_total.cost or 0.0) + (usage.cost or 0.0)

            candidate = self._strip_fences(raw)
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj, usage_total, raw
            except Exception:
                pass

            # If it looks truncated, ask the model to re-output a complete JSON object.
            if attempt < retries:
                truncated = not self._json_complete_heuristic(candidate)
                fixup = (
                    "Твой предыдущий JSON был неполным/обрезанным. "
                    "Верни ПОЛНЫЙ валидный JSON-объект целиком (без ```), начиная с '{' и заканчивая '}'."
                    if truncated
                    else "Верни валидный JSON-объект целиком (без ```), начиная с '{' и заканчивая '}'."
                )
                attempt_messages = attempt_messages + [{"role": "user", "content": fixup}]

        return {}, usage_total, last_raw

    @staticmethod
    def _normalize_options(options: Any) -> dict[str, str]:
        normalized: dict[str, str] = {}
        if isinstance(options, dict):
            for key, value in options.items():
                label = str(key).strip()
                text = str(value).strip()
                if label and text:
                    normalized[label] = text
            return normalized
        if isinstance(options, list):
            for index, item in enumerate(options, start=1):
                if isinstance(item, dict):
                    label = str(item.get("label") or index).strip()
                    text = str(item.get("text") or "").strip()
                else:
                    label = str(index)
                    text = str(item).strip()
                if label and text:
                    normalized[label] = text
        return normalized

    @staticmethod
    def _parse_question_and_options_from_text(text: str) -> dict[str, Any]:
        cleaned = OpenRouterClient._strip_fences(text)
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        option_re = re.compile(r"^(?P<label>(?:[1-9]|[A-DА-Г]))[\)\.\:-]?\s+(?P<text>.+)$", re.IGNORECASE)
        options: dict[str, str] = {}
        question_lines: list[str] = []
        seen_option = False
        for line in lines:
            match = option_re.match(line)
            if match:
                seen_option = True
                label = match.group("label").upper()
                options[label] = match.group("text").strip()
                continue
            if seen_option and options:
                last_key = next(reversed(options))
                options[last_key] = f"{options[last_key]} {line}".strip()
            else:
                question_lines.append(line)
        question = " ".join(question_lines).strip()
        return {"question": question, "options": options}

    def _extract_json_question_options(
        self, image_bytes: bytes, mime: str, model: str, *, max_tokens: int = 1400
    ) -> tuple[dict[str, Any], OpenRouterUsage, str]:
        prompt = (
            "Извлеки из изображения только текст задания (вопроса) и варианты ответа.\n"
            "Игнорируй шапку сайта, меню, номера страниц, кнопки и прочий интерфейс.\n"
            "Задание может НЕ содержать '?' (например, заканчивается ':') — всё равно верни это как question.\n"
            "Поддерживай варианты с метками 1,2,3,4... и А,Б,В,Г / A,B,C,D.\n"
            "Если варианты без меток (радиокнопки), верни их как \"1\",\"2\",\"3\"...\n"
            "Верни строго JSON:\n"
            "{\n"
            '  "question": "полный текст вопроса",\n'
            '  "options": {\n'
            '    "1": "текст варианта 1",\n'
            '    "2": "текст варианта 2"\n'
            "  }\n"
            "}\n"
            'Если вариантов не видно, верни пустой объект "options": {}.\n'
            'Если вопрос не виден, верни {"question":"","options":{}}.'
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _data_url(image_bytes, mime)}},
                ],
            }
        ]
        data, usage, raw = self._call_json_object_with_retry(
            model=model,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=0.1,
            retries=2,
        )
        question = str((data or {}).get("question") or "").strip()
        options = self._normalize_options((data or {}).get("options"))
        return {"question": question, "options": options}, usage, raw

    def _extract_raw_text(self, image_bytes: bytes, mime: str, model: str) -> tuple[str, OpenRouterUsage]:
        prompt = (
            "Прочитай текст на изображении максимально близко к оригиналу.\n"
            "Игнорируй элементы интерфейса сайта, если они не относятся к самому вопросу.\n"
            "Верни только распознанный текст без пояснений."
        )
        payload = {
            "model": model,
            "max_tokens": 1200,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _data_url(image_bytes, mime)}},
                    ],
                }
            ],
        }
        resp = self._post(payload)
        return self._extract_text(resp), self._extract_usage(resp)

    def ocr_page_to_markdown(self, image_bytes: bytes, mime: str, model: str) -> tuple[str, OpenRouterUsage]:
        prompt = (
            "OCR this page into Markdown.\n"
            "- Output Markdown only, no preamble.\n"
            "- Do NOT include image links or URLs.\n"
            "- Keep tables as Markdown tables when possible.\n"
            "- If there are anatomical diagrams/figures, add a 1–2 sentence description as plain text.\n"
        )
        payload = {
            "model": model,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _data_url(image_bytes, mime)}},
                    ],
                }
            ],
        }
        resp = self._post(payload)
        return self._extract_text(resp), self._extract_usage(resp)

    def extract_question_from_screenshot(self, image_bytes: bytes, mime: str, model: str) -> tuple[str, OpenRouterUsage]:
        prompt = (
            "Extract only the exam question from this screenshot in Russian.\n"
            "Do not answer it. Ignore UI chrome, watermarks, timers, and unrelated text.\n"
            "If there is no clear question or the selected region is wrong, return exactly: NO_QUESTION_FOUND"
        )
        payload = {
            "model": model,
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _data_url(image_bytes, mime)}},
                    ],
                }
            ],
        }
        resp = self._post(payload)
        return self._extract_text(resp), self._extract_usage(resp)

    def grounded_answer(self, question: str, sources_text: str, model: str, *, max_tokens: int = 900) -> tuple[str, OpenRouterUsage]:
        payload = {
            "model": model,
            "max_tokens": int(max_tokens),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты осторожный помощник. Отвечай ТОЛЬКО по приведённым источникам из учебника. "
                        "Если в источниках нет прямого подтверждения, напиши ровно: "
                        "\"Не найдено в учебнике по предоставленным источникам.\" "
                        "Не додумывай и не подменяй тему. Пиши по-русски. "
                        "Обязательно приведи минимум 1 короткую цитату (до 20 слов) и укажи страницу в формате (стр. N). "
                        "Если не можешь привести цитату и страницу — верни строку \"Не найдено в учебнике по предоставленным источникам.\""
                    ),
                },
                {"role": "user", "content": f"Question: {question}\n\nSources:\n{sources_text}"},
            ],
        }
        resp = self._post(payload)
        return self._extract_text(resp), self._extract_usage(resp)

    def reformulate_for_retrieval(self, question: str, model: str, *, max_tokens: int = 220) -> tuple[str, OpenRouterUsage]:
        payload = {
            "model": model,
            "max_tokens": int(max_tokens),
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты помощник для поиска по учебнику. "
                        "Переформулируй вопрос в короткий поисковый запрос (1 строка) "
                        "по-русски, без воды, только ключевые термины."
                    ),
                },
                {"role": "user", "content": f"Вопрос: {question}"},
            ],
        }
        resp = self._post(payload)
        return self._extract_text(resp).strip(), self._extract_usage(resp)

    def judge_retrieval_relevance(
        self,
        question: str,
        retrieved_text: str,
        model: str,
        *,
        max_tokens: int = 450,
        retries: int = 1,
    ) -> tuple[dict[str, Any], OpenRouterUsage]:
        # Avoid JSON here: some models/providers return truncated JSON even with response_format=json_object.
        # Use a strict single-line format and parse with regex.
        import re

        system_prompt = (
            "Ты — валидатор RAG. Оцени, относится ли фрагмент учебника к вопросу.\n"
            "Ответь ОДНОЙ строкой строго в формате:\n"
            "score=<0..1>; relevant=<yes/no>; reason=<коротко>\n"
            "Никаких JSON, никаких переносов строк."
        )
        user_prompt = f"Вопрос:\n{question}\n\nФрагмент:\n{retrieved_text}"

        payload = {
            "model": model,
            "max_tokens": int(max_tokens),
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        usage_total = OpenRouterUsage()
        last_raw = ""
        for _attempt in range(max(1, int(retries) + 1)):
            resp = self._post(payload)
            raw = self._extract_text(resp)
            last_raw = raw
            usage = self._extract_usage(resp)
            usage_total.prompt_tokens = (usage_total.prompt_tokens or 0) + (usage.prompt_tokens or 0)
            usage_total.completion_tokens = (usage_total.completion_tokens or 0) + (usage.completion_tokens or 0)
            usage_total.total_tokens = (usage_total.total_tokens or 0) + (usage.total_tokens or 0)
            usage_total.cost = (usage_total.cost or 0.0) + (usage.cost or 0.0)

            line = " ".join(self._strip_fences(raw).splitlines()).strip()
            m_score = re.search(r"score\s*=\s*([01](?:\.\d+)?)", line, flags=re.IGNORECASE)
            m_rel = re.search(r"relevant\s*=\s*(yes|no|true|false)", line, flags=re.IGNORECASE)
            m_reason = re.search(r"reason\s*=\s*(.+)$", line, flags=re.IGNORECASE)
            if m_score and m_rel:
                try:
                    score = float(m_score.group(1))
                except Exception:
                    score = 0.0
                rel_raw = m_rel.group(1).lower()
                is_relevant = rel_raw in ("yes", "true")
                reason = (m_reason.group(1).strip() if m_reason else "").strip()
                return {"score": score, "is_relevant": is_relevant, "reason": reason}, usage_total

            payload["messages"] = payload["messages"] + [
                {"role": "user", "content": "Повтори ровно одной строкой: score=<0..1>; relevant=<yes/no>; reason=<...>."}
            ]

        return {"score": 0.0, "is_relevant": False, "reason": f"Ошибка парсинга: {self._strip_fences(last_raw)[:200]}"}, usage_total

    def compare_multiple_choice(
        self,
        question: str,
        options: dict[str, str],
        sources_text: str,
        model: str,
        *,
        max_tokens: int = 2000,
        retries: int = 1,
    ) -> tuple[dict[str, Any], OpenRouterUsage]:
        """Сравнивает варианты ответов с учебником."""
        options_text = "\n".join([f"{letter}) {text}" for letter, text in options.items()])

        system_prompt = (
            "Ты — экзаменатор по физиологии. "
            "Твоя задача — определить ПРАВИЛЬНЫЙ вариант ответа на основе ТОЛЬКО приведённого текста учебника. "
            "Если в учебнике нет точного ответа, укажи confidence < 0.5. "
            "Не используй JSON. Отвечай строго построчно по шаблону."
        )

        user_prompt = (
            f"Вопрос: {question}\n\n"
            f"Варианты ответов:\n{options_text}\n\n"
            f"Текст из учебника:\n{sources_text}\n\n"
            "Ответь строго так:\n"
            "CORRECT_OPTION: <буква или ?>\n"
            "CONFIDENCE: <число от 0 до 1>\n"
            "CORRECT_TEXT: <текст правильного варианта>\n"
            "REASONING: <краткое объяснение на основе источников>"
        )

        payload = {
            "model": model,
            "max_tokens": int(max_tokens),
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        usage_total = OpenRouterUsage()
        last_raw = ""
        for _attempt in range(max(1, int(retries) + 1)):
            resp = self._post(payload)
            raw = self._extract_text(resp)
            last_raw = raw
            usage = self._extract_usage(resp)
            usage_total.prompt_tokens = (usage_total.prompt_tokens or 0) + (usage.prompt_tokens or 0)
            usage_total.completion_tokens = (usage_total.completion_tokens or 0) + (usage.completion_tokens or 0)
            usage_total.total_tokens = (usage_total.total_tokens or 0) + (usage.total_tokens or 0)
            usage_total.cost = (usage_total.cost or 0.0) + (usage.cost or 0.0)

            text = self._strip_fences(raw)
            data = self._parse_multiple_choice_text(text, options)
            if data.get("correct_option") != "?":
                return data, usage_total

            payload["messages"] = payload["messages"] + [
                {
                    "role": "user",
                    "content": (
                        "Повтори строго построчно: CORRECT_OPTION, CONFIDENCE, "
                        "CORRECT_TEXT, REASONING. Без JSON."
                    ),
                }
            ]

        data = self._parse_multiple_choice_text(self._strip_fences(last_raw), options)
        if data.get("correct_option") == "?":
            data["reasoning"] = f"Ошибка парсинга: {self._strip_fences(last_raw)[:200]}"
        return data, usage_total

    @staticmethod
    def _parse_multiple_choice_text(text: str, options: dict[str, str]) -> dict[str, Any]:
        correct_option = "?"
        confidence = 0.0
        correct_text = ""
        reasoning = ""

        normalized = text.strip()
        match_option = re.search(r"CORRECT_OPTION\s*:\s*([0-9A-ZА-Я])", normalized, flags=re.IGNORECASE)
        if not match_option:
            match_option = re.search(r'"correct_option"\s*:\s*"([0-9A-ZА-Я])"', normalized, flags=re.IGNORECASE)
        if match_option:
            correct_option = match_option.group(1).upper()

        match_conf = re.search(r"CONFIDENCE\s*:\s*([01](?:\.\d+)?)", normalized, flags=re.IGNORECASE)
        if not match_conf:
            match_conf = re.search(r'"confidence"\s*:\s*([01](?:\.\d+)?)', normalized, flags=re.IGNORECASE)
        if match_conf:
            try:
                confidence = float(match_conf.group(1))
            except Exception:
                confidence = 0.0

        match_text = re.search(
            r"CORRECT_TEXT\s*:\s*(.*?)(?:\n\s*REASONING\s*:|\Z)",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match_text:
            correct_text = match_text.group(1).strip()

        match_reason = re.search(r"REASONING\s*:\s*(.*)\Z", normalized, flags=re.IGNORECASE | re.DOTALL)
        if match_reason:
            reasoning = match_reason.group(1).strip()

        if correct_option in options and not correct_text:
            correct_text = options[correct_option]
        return {
            "correct_option": correct_option,
            "confidence": confidence,
            "reasoning": reasoning,
            "correct_text": correct_text,
        }

    def extract_question_and_options(
        self,
        image_bytes: bytes,
        mime: str,
        model: str,
        *,
        max_tokens: int = 1400,
    ) -> tuple[dict[str, Any], OpenRouterUsage, str]:
        """Извлекает вопрос и варианты ответов из скриншота."""
        models_to_try = [model]

        total_usage = OpenRouterUsage()
        raw_debug_parts: list[str] = []
        for candidate in models_to_try:
            parsed, usage, raw = self._extract_json_question_options(image_bytes, mime, candidate, max_tokens=int(max_tokens))
            raw_debug_parts.append(f"[json:{candidate}]\n{raw}")
            total_usage.prompt_tokens = (total_usage.prompt_tokens or 0) + (usage.prompt_tokens or 0)
            total_usage.completion_tokens = (total_usage.completion_tokens or 0) + (usage.completion_tokens or 0)
            total_usage.total_tokens = (total_usage.total_tokens or 0) + (usage.total_tokens or 0)
            total_usage.cost = (total_usage.cost or 0.0) + (usage.cost or 0.0)
            if parsed.get("question"):
                return parsed, total_usage, "\n\n".join(raw_debug_parts)

            raw_text, usage2 = self._extract_raw_text(image_bytes, mime, candidate)
            raw_debug_parts.append(f"[raw:{candidate}]\n{raw_text}")
            total_usage.prompt_tokens = (total_usage.prompt_tokens or 0) + (usage2.prompt_tokens or 0)
            total_usage.completion_tokens = (total_usage.completion_tokens or 0) + (usage2.completion_tokens or 0)
            total_usage.total_tokens = (total_usage.total_tokens or 0) + (usage2.total_tokens or 0)
            total_usage.cost = (total_usage.cost or 0.0) + (usage2.cost or 0.0)
            parsed_from_text = self._parse_question_and_options_from_text(raw_text)
            if parsed_from_text.get("question"):
                return parsed_from_text, total_usage, "\n\n".join(raw_debug_parts)

        return {"question": "", "options": {}}, total_usage, "\n\n".join(raw_debug_parts)
