from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

from src.log import emit as log
from src.models import ProviderConfig, Script, Section
from src.providers.base import LLMProvider
from src.providers.wiro_client import WiroClient


def _repair_truncated_json(text: str) -> str:
    """Attempt to repair truncated JSON by closing open strings, arrays, objects."""
    # If we're inside an unterminated string, close it
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
    if in_string:
        text += '"'

    # Remove any trailing partial key-value (e.g. `"key": "partial`)
    # by stripping from the last complete value
    text = text.rstrip()
    if text.endswith(","):
        text = text[:-1]

    # Count open braces/brackets and close them
    opens = {"[": 0, "{": 0}
    closes = {"]": "[", "}": "{"}
    in_str = False
    esc = False
    for ch in text:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in opens:
            opens[ch] += 1
        elif ch in closes:
            opens[closes[ch]] -= 1

    # Close in reverse order of what's open
    text += "]" * opens["["]
    text += "}" * opens["{"]
    return text


def _extract_json(raw: str) -> dict:
    """Extract and parse a JSON object from potentially messy LLM output."""
    text = raw.strip()

    # Remove markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)

    # Find the first {
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output.")

    # Try to find the matching closing brace
    depth = 0
    end = -1
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end != -1:
        json_str = text[start : end + 1]
    else:
        # Truncated output — attempt repair
        json_str = _repair_truncated_json(text[start:])

    # Remove trailing commas before } or ]
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    return json.loads(json_str)


RUN_URL = "https://api.wiro.ai/v1/Run/google/gemini-3-pro"

SYSTEM_PROMPT = """\
You are an engaging YouTube script writer. Output ONLY valid JSON, nothing else.
Write narration that sounds natural when read aloud — conversational, vivid, and entertaining.

For image prompts, use the Nano Banana 2 ULTIMATE PROMPT CHEATSHEET formula:
[TYPE] + [SUBJECT] + [ACTION] + [SETTING] + [LIGHT] + [CAMERA] + [STYLE] + [TEXTURE] + [COLOR] + [OUTPUT]

TYPE categories: photo, illustration, digital art, 3D render, painting, sketch, cartoon, anime, comic, infographic
SUBJECT: main focus (person, object, scene)
ACTION: what the subject is doing
SETTING: location/environment
LIGHT: lighting style (cinematic, soft, dramatic, natural, neon, etc.)
CAMERA: angle/distance (close-up, wide shot, aerial, etc.)
STYLE: artistic style (realistic, abstract, minimalist, etc.)
TEXTURE: surface quality (smooth, rough, metallic, etc.)
COLOR: color scheme (vibrant, monochrome, pastel, etc.)
OUTPUT: final format (16:9, 4K, high detail, etc.)

Each image prompt should be ONE detailed sentence following this formula."""

USER_PROMPT_TEMPLATE = """\
Create a listicle script about: "{topic}"
{style_instruction}
Return ONLY this JSON (no other text):
{{
  "title": "catchy video title",
  "intro_narration": "4-5 sentences that hook the viewer, set the stage, and tease what's coming",
  "intro_image_prompt": "Digital art YouTube thumbnail with bold text overlay reading the video title, vibrant colors, dramatic lighting, close-up shot, realistic style, smooth texture, high contrast, 16:9, high detail",
  "sections": [
    {{
      "number": 1,
      "heading": "section heading",
      "narration": "Start with 'Number One, heading!' then 5-7 sentences with interesting facts, details, and smooth transitions",
      "image_prompts": [
        "Photo of section heading text overlay with subject performing action, professional setting, cinematic lighting, medium shot, realistic style, sharp texture, vibrant colors, 16:9, high detail",
        "Illustration of same subject from different angle performing action, detailed background, soft lighting, wide shot, artistic style, smooth texture, complementary colors, 16:9, high detail"
      ]
    }}
  ],
  "outro_narration": "2-3 sentences wrapping up with a call to action",
  "outro_image_prompt": "3D render of video topic scene with semi-transparent subscribe button and like button in corner, professional setting, dramatic lighting, medium shot, realistic style, smooth texture, warm colors, 16:9, high detail"
}}

Rules:
- Exactly {num_sections} sections
- intro_narration: 4-5 engaging sentences that hook the viewer
- intro_image_prompt: must follow Nano Banana 2 formula with YouTube title as text overlay
- Each section narration MUST begin with announcing the number and heading like "Number One, FlowState AI!" or "Number Three, CodeWhisper Pro!" followed by 5-7 detailed sentences, conversational tone
- image_prompts: REQUIRED array of exactly {images_per_section} prompts per section
  - ALL prompts MUST follow the Nano Banana 2 formula: [TYPE] + [SUBJECT] + [ACTION] + [SETTING] + [LIGHT] + [CAMERA] + [STYLE] + [TEXTURE] + [COLOR] + [OUTPUT]
  - The FIRST prompt MUST include bold readable text overlay of the section heading
  - All OTHER prompts must be different angles, views, or perspectives of the same subject with NO TEXT, purely visual imagery
  - Each prompt must be unique and strictly relevant to the section topic
  - Use specific TYPE categories: photo, illustration, digital art, 3D render, painting, sketch, cartoon, anime, comic, infographic
- outro_narration: 2-3 sentences with call to action
- outro_image_prompt: must follow Nano Banana 2 formula with subscribe and like buttons overlaid naturally in a corner — NOT a generic end screen
- All narration should sound natural when spoken aloud"""

SUBTITLES_ADDENDUM = """

You MUST use these exact section headings (in this order):
{subtitles}
Do NOT rename or reorder them. Write narration and image prompts for each."""

MAX_RETRIES = 2


class WiroLLMProvider(LLMProvider):
    """Wiro-based LLM provider using google/gemini-3-flash."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.client = WiroClient(config)

    async def _fetch_content(self, payload: dict) -> str:
        """Submit task, poll, and return raw text content."""
        task = await self.client.run_and_poll(RUN_URL, payload)
        urls = WiroClient.get_output_urls(task)
        if not urls:
            debug = task.get("debugoutput", "")
            if debug:
                return debug
            raise RuntimeError("No output from Wiro LLM task.")

        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.get(urls[0])
            resp.raise_for_status()
            return resp.text

    async def generate_script(
        self, topic: str, num_sections: int, subtitles: list[str] | None = None,
        image_style: str = "", images_per_section: int = 1,
    ) -> Script:
        style_instruction = ""
        if image_style:
            style_instruction = f'\nAll image prompts must be in "{image_style}" style. Append ", {image_style} style" to every image_prompt.\n'
        user_prompt = USER_PROMPT_TEMPLATE.format(
            topic=topic, num_sections=num_sections,
            style_instruction=style_instruction,
            images_per_section=max(1, images_per_section),
        )
        if subtitles:
            numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(subtitles))
            user_prompt += SUBTITLES_ADDENDUM.format(subtitles=numbered)
            # Override section count to match subtitles
            num_sections = len(subtitles)

        payload = {
            "prompt": user_prompt,
            "systemInstructions": SYSTEM_PROMPT,
            "thinkingLevel": "low",
        }

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                log(f"llm attempt {attempt + 1}/{MAX_RETRIES + 1}")
                content = await self._fetch_content(payload)

                # Save raw output for debugging
                debug_dir = Path("output") / "_debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / f"llm_raw_{attempt}.txt").write_text(
                    content, encoding="utf-8"
                )

                log(f"llm raw output: {len(content)} chars")
                data = _extract_json(content)
                # Strip whitespace from top-level keys
                data = {k.strip(): v for k, v in data.items()}

                raw_sections = data.get("sections", [])
                if not raw_sections:
                    raise ValueError(f"No sections in LLM output (keys: {list(data.keys())})")

                sections = []
                for raw_s in raw_sections:
                    # Strip whitespace from keys (LLM sometimes adds trailing spaces)
                    s = {k.strip(): v for k, v in raw_s.items()}
                    if not s.get("heading"):
                        s["heading"] = f"Section {s.get('number', len(sections) + 1)}"
                    # LLM sometimes uses variant key names for narration
                    if not s.get("narration"):
                        for alt in ("script", "text", "content", "description", "body"):
                            if s.get(alt):
                                s["narration"] = s.pop(alt)
                                break
                        else:
                            s["narration"] = f"Let's talk about {s.get('heading', 'this topic')}."
                    # Handle image_prompts array or fallback from singular image_prompt
                    prompts = s.get("image_prompts", [])
                    if isinstance(prompts, str):
                        prompts = [prompts]
                    if not prompts and s.get("image_prompt"):
                        prompts = [s["image_prompt"]]
                    if not prompts:
                        heading = s.get("heading", "")
                        prompts = [
                            f"Cinematic illustration of: {heading}, "
                            f"with bold text overlay reading '{heading}', "
                            f"dramatic lighting, high detail, 16:9"
                        ]
                    sections.append(Section(
                        number=s.get("number", len(sections) + 1),
                        heading=s["heading"],
                        narration=s["narration"],
                        image_prompt=prompts[0],
                        image_prompts=prompts,
                    ))

                outro = data.get("outro_narration", "")
                if not outro or len(outro) < 20:
                    outro = (
                        f"And that wraps up our list! If you enjoyed this video, "
                        f"make sure to like, subscribe, and hit that notification bell "
                        f"so you never miss out on our next one. See you in the next video!"
                    )

                intro_img = data.get("intro_image_prompt", "")
                if not intro_img:
                    intro_img = f"Cinematic wide shot representing: {data.get('title', topic)}, dramatic lighting, high detail, 16:9"
                outro_img = data.get("outro_image_prompt", "")
                if not outro_img:
                    outro_img = f"Cinematic closing shot for a video about: {data.get('title', topic)}, warm lighting, high detail, 16:9"

                return Script(
                    title=data.get("title", topic),
                    intro_narration=data.get("intro_narration", ""),
                    intro_image_prompt=intro_img,
                    sections=sections,
                    outro_narration=outro,
                    outro_image_prompt=outro_img,
                )

            except Exception as e:
                last_error = e
                log(f"llm attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES:
                    continue

        raise RuntimeError(
            f"Script generation failed after {MAX_RETRIES + 1} attempts: {last_error}"
        )
