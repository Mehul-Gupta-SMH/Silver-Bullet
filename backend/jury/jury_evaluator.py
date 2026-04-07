"""LLM-as-jury evaluator for SilverBullet.

Replaces the CNN inference path with a structured LLM judge that asks a battery
of binary yes/no questions mirroring the CNN's feature clusters.

Clusters → questions
────────────────────
Semantic (PREC)       : Is text2 semantically grounded in text1?
Lexical (dice, rouge3): Does text2 use substantially similar wording to text1?
NLI entailment        : Does text1 entail text2?
NLI contradiction     : Does text2 contradict text1? (inverted for score)
Entity consistency    : Are all named entities in text2 consistent with text1?
LCS                   : Does text2 preserve the key sequence of facts/steps from text1?

Each question gets { answer, confidence, reasoning }.
Final score = weighted mean of (answer=="yes" ? confidence : 1-confidence).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI, OpenAIError

from backend.api.schemas import EvaluationMode, JuryQuestion, JuryResult
from backend.resources.getConfig import getVal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Question definitions
# Each entry: (cluster_id, question_text, weight, higher_is_faithful)
#   higher_is_faithful=False  → a "yes" answer means more hallucination
#   score contribution = confidence if yes & higher_is_faithful, else 1-confidence
# ---------------------------------------------------------------------------

_QUESTIONS: list[tuple[str, str, float, bool]] = [
    (
        "semantic",
        "Is text2 semantically grounded in text1? (yes/no)",
        1.0,
        True,
    ),
    (
        "lexical",
        "Does text2 use substantially similar wording to text1? (yes/no)",
        1.0,
        True,
    ),
    (
        "nli_entailment",
        "Does text1 entail text2? (yes/no)",
        1.0,
        True,
    ),
    (
        "nli_contradiction",
        "Does text2 contradict text1? (yes/no)",
        1.0,
        False,   # "yes" → contradiction → unfaithful
    ),
    (
        "entity",
        (
            "Are all named entities (locations, products, laws, times, durations, "
            "percentages) in text2 consistent with text1? (yes/no)"
        ),
        1.0,
        True,
    ),
    (
        "lcs",
        "Does text2 preserve the key sequence of facts/steps from text1? (yes/no)",
        1.0,
        True,
    ),
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precision factual-consistency judge for an LLM evaluation pipeline.
You will be given two texts (TEXT1 and TEXT2) and an evaluation mode.

Evaluation modes:
- context-vs-generated : TEXT1 is the source context, TEXT2 is an LLM-generated answer.
  Check whether TEXT2 is grounded in TEXT1 without hallucinations.
- reference-vs-generated : TEXT1 is the ground-truth reference, TEXT2 is a generated answer.
  Check whether TEXT2 faithfully captures the reference.
- model-vs-model : TEXT1 and TEXT2 are outputs from two different LLMs for the same prompt.
  Check whether they agree with each other.

You must answer each question below with a JSON object.
Respond ONLY with valid JSON — no markdown, no preamble, no trailing text.

For each question provide:
  "answer"     : "yes" or "no" (lowercase, exactly one word)
  "confidence" : float in [0.0, 1.0] — how confident you are in the answer
  "reasoning"  : one concise sentence explaining your answer

Use the following diagnostic codes in your reasoning where relevant:
  HALL-NUMERIC  — text2 contains a wrong number not present in text1
  ENT-SUBST     — text2 substitutes a named entity for a different one
  NEG-FACT      — text2 negates a fact that text1 asserts
  OMIT-KEY      — text2 omits a key claim or step from text1
  FAITHFUL      — text2 is grounded and consistent with text1

Required JSON shape (keys must match exactly):
{
  "answers": [
    {
      "cluster_id": "<cluster_id>",
      "answer": "yes" | "no",
      "confidence": <float>,
      "reasoning": "<one sentence>"
    },
    ...
  ]
}
"""

_USER_TEMPLATE = """\
Evaluation mode: {mode}

TEXT1:
{text1}

TEXT2:
{text2}

Questions (answer in the same order as listed):
{questions}
"""


def _build_question_block() -> str:
    lines = []
    for i, (cluster_id, question, _weight, _faithful) in enumerate(_QUESTIONS, 1):
        lines.append(f'{i}. cluster_id="{cluster_id}" — {question}')
    return "\n".join(lines)


_QUESTION_BLOCK = _build_question_block()


# ---------------------------------------------------------------------------
# JuryEvaluator
# ---------------------------------------------------------------------------

class JuryEvaluator:
    """Evaluate a text pair using an LLM as a structured binary-question jury.

    Args:
        model: OpenAI model name.  Defaults to SB_JURY_MODEL env var or "gpt-4o-mini".
    """

    def __init__(self, model: str | None = None) -> None:
        config = getVal(env="DEVELOPMENT")
        api_key: str | None = config.get("openai_token") or os.environ.get("SB_OPENAI_TOKEN")
        if not api_key:
            raise ValueError(
                "Jury mode requires an OpenAI API key. "
                "Set SB_OPENAI_TOKEN or add openai_token to config.yaml."
            )
        self._client = OpenAI(api_key=api_key)
        self._model: str = (
            model
            or os.environ.get("SB_JURY_MODEL", "gpt-4o-mini")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, text1: str, text2: str, mode: EvaluationMode) -> JuryResult:
        """Run the jury prompt for a single pair and return a :class:`JuryResult`.

        Args:
            text1: First text (source / reference / context).
            text2: Second text (hypothesis / generated output).
            mode:  Evaluation mode string.

        Returns:
            JuryResult with aggregated score, verdict, per-question breakdown, and model name.

        Raises:
            RuntimeError: If the LLM call fails or returns unparseable JSON.
        """
        raw = self._call_llm(text1, text2, mode)
        questions = self._parse_response(raw)
        score = self._aggregate(questions)
        verdict: str = "faithful" if score >= 0.5 else "hallucinated"
        return JuryResult(
            score=round(score, 4),
            verdict=verdict,  # type: ignore[arg-type]
            questions=questions,
            model_used=self._model,
        )

    def evaluate_batch(
        self,
        pairs: list[tuple[str, str, EvaluationMode]],
    ) -> list[JuryResult]:
        """Evaluate a list of (text1, text2, mode) triples sequentially.

        Args:
            pairs: List of (text1, text2, mode) tuples.

        Returns:
            List of JuryResult objects in the same order as *pairs*.
        """
        results: list[JuryResult] = []
        for text1, text2, mode in pairs:
            results.append(self.evaluate(text1, text2, mode))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, text1: str, text2: str, mode: str) -> str:
        """Send the structured prompt to the LLM and return the raw response text.

        Raises:
            RuntimeError: On OpenAI API errors.
        """
        user_message = _USER_TEMPLATE.format(
            mode=mode,
            text1=text1,
            text2=text2,
            questions=_QUESTION_BLOCK,
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
            )
        except OpenAIError as exc:
            logger.error("OpenAI API call failed: %s", exc)
            raise RuntimeError(f"LLM jury call failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned an empty response.")
        return content

    def _parse_response(self, raw: str) -> list[JuryQuestion]:
        """Parse the LLM JSON response into a list of :class:`JuryQuestion`.

        Validates that each expected cluster_id is present and that answer/confidence
        values are within valid ranges.

        Raises:
            RuntimeError: If JSON is malformed or required fields are missing.
        """
        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned non-JSON output: {exc}\nRaw: {raw[:500]}") from exc

        if not isinstance(data, dict) or "answers" not in data:
            raise RuntimeError(
                f"LLM response missing 'answers' key. Got keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
            )

        answers: list[Any] = data["answers"]
        if not isinstance(answers, list) or len(answers) == 0:
            raise RuntimeError("LLM 'answers' field is empty or not a list.")

        # Index the LLM answers by cluster_id for robust lookup
        answer_map: dict[str, Any] = {}
        for item in answers:
            if isinstance(item, dict) and "cluster_id" in item:
                answer_map[item["cluster_id"]] = item

        questions: list[JuryQuestion] = []
        for cluster_id, question_text, _weight, _faithful in _QUESTIONS:
            item = answer_map.get(cluster_id)
            if item is None:
                # Fall back to positional lookup if the LLM omitted cluster_id
                idx = [q[0] for q in _QUESTIONS].index(cluster_id)
                item = answers[idx] if idx < len(answers) else {}

            raw_answer = str(item.get("answer", "no")).strip().lower()
            answer_val = raw_answer if raw_answer in ("yes", "no") else "no"

            raw_conf = item.get("confidence", 0.5)
            try:
                confidence = float(raw_conf)
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))

            reasoning = str(item.get("reasoning", "")).strip()

            questions.append(
                JuryQuestion(
                    question=question_text,
                    answer=answer_val,  # type: ignore[arg-type]
                    confidence=confidence,
                    reasoning=reasoning,
                )
            )

        return questions

    def _aggregate(self, questions: list[JuryQuestion]) -> float:
        """Compute the weighted faithfulness score from the parsed jury questions.

        For questions where a "yes" answer indicates faithfulness
        (higher_is_faithful=True):
            contribution = confidence if answer=="yes" else 1 - confidence

        For questions where a "yes" answer indicates hallucination
        (higher_is_faithful=False, e.g. contradiction):
            contribution = 1 - confidence if answer=="yes" else confidence

        Returns:
            Weighted mean in [0, 1].  Higher = more faithful.
        """
        if not questions:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for q, (cluster_id, _question_text, weight, higher_is_faithful) in zip(
            questions, _QUESTIONS
        ):
            if higher_is_faithful:
                contribution = q.confidence if q.answer == "yes" else 1.0 - q.confidence
            else:
                # "yes" means bad (e.g. contradiction detected)
                contribution = 1.0 - q.confidence if q.answer == "yes" else q.confidence

            weighted_sum += weight * contribution
            total_weight += weight

        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight
