"""Arabic diacritization pipeline using the Ad-dabit-fadel model.

Pipeline stages:
    1. Tokenization        : split text into base letters + [MASK] placeholders.
    2. Inference           : single forward pass OR sliding-window majority vote
                             when the tokenized sequence exceeds the model limit.
    3. Reconstruction      : map predicted label ids back to harakat characters.

Public API:
    diacritic_text(text)   : diacritize a string (used by the FastAPI route).
    get_pipeline()         : singleton pipeline instance (FastAPI dependency-friendly).
"""

from __future__ import annotations

import logging
import pickle
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

logger = logging.getLogger(__name__)


# --- Configuration -----------------------------------------------------------

MODEL_NAME = "AbderrahmanSkiredj1/Ad-dabit-fadel"
CONST_DIR = Path(__file__).parent / "const"

MODEL_MAX_TOKENS = 512
WINDOW_MAX_TOKENS = 480  # leave headroom for [CLS], [SEP], sub-word splits
WINDOW_STEP = 10         # Ad-dabit(10): slide by 10 word-units

HARAKAT_MAPPING: Dict[str, str] = {
    "فتحة": "\u064e", "فتحتان": "\u064b", "ضمة": "\u064f", "ضمتان": "\u064c",
    "كسرة": "\u0650", "كسرتان": "\u064d", "سكون": "\u0652", "شدة": "\u0651",
    "شدة فتحة": "\u0651\u064e", "شدة فتحتان": "\u0651\u064b", "شدة ضمة": "\u0651\u064f",
    "شدة ضمتان": "\u0651\u064c", "شدة كسرة": "\u0651\u0650", "شدة كسرتان": "\u0651\u064d",
    "تطويل": "\u0640", "X": "",
}
REV_HARAKAT_MAPPING: Dict[str, str] = {v: k for k, v in HARAKAT_MAPPING.items()}
RE_DIACRITICS = re.compile(r"[\u064b-\u0652\u0670]")


def _load_pickle(filename: str):
    path = CONST_DIR / filename
    with path.open("rb") as f:
        return pickle.load(f)


# --- Pipeline ----------------------------------------------------------------

class AdDabitDiacritizationPipeline:
    """Arabic diacritization pipeline with sliding-window majority voting."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading diacritization model '%s' on %s", model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.label2id = self.model.config.label2id
        self.id2label = self.model.config.id2label

        # Kept to preserve parity with the original pipeline's asset loading.
        self.arabic_letters = _load_pickle("ARABIC_LETTERS_LIST.pickle")
        self.diacritics_list = _load_pickle("DIACRITICS_LIST.pickle")
        self.characters_mapping = _load_pickle("CHARACTERS_MAPPING.pickle")

    # -- Stage 1: tokenization ------------------------------------------------

    def _tokenize(
        self, text: str
    ) -> Tuple[List[str], List[int], List[Tuple[str, int]]]:
        """Split text into ``[word, MASK, MASK, ...]`` tokens per word."""
        input_tokens: List[str] = []
        truth_tokens: List[int] = []
        word_info: List[Tuple[str, int]] = []
        text = self.filter_arabic_only(text)

        for word in text.split():
            base_word = ""
            harakats: List[int] = []

            i = 0
            while i < len(word):
                char = word[i]
                if RE_DIACRITICS.match(char):
                    # Stray diacritic at word start — skip.
                    i += 1
                    continue

                base_word += char
                temp = ""
                j = i + 1
                while j < len(word) and RE_DIACRITICS.match(word[j]):
                    temp += word[j]
                    j += 1

                label = REV_HARAKAT_MAPPING.get(temp, "X") if temp else "X"
                harakats.append(self.label2id[label])
                i = j

            input_tokens.append(base_word)
            input_tokens.extend(["[MASK]"] * len(harakats))
            truth_tokens.append(-100)  # base word is not scored
            truth_tokens.extend(harakats)
            word_info.append((base_word, len(base_word)))

        return input_tokens, truth_tokens, word_info

    # -- Stage 2: inference ---------------------------------------------------

    @torch.inference_mode()
    def _forward(self, input_tokens: List[str]) -> Tuple[List[int], List[int | None]]:
        inputs = self.tokenizer(
            input_tokens, is_split_into_words=True, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)[0].tolist()
        return predictions, inputs.word_ids()

    @staticmethod
    def _group_word_units(input_tokens: List[str]) -> List[List[str]]:
        """Group tokens into ``[word, MASK, MASK, ...]`` units."""
        units: List[List[str]] = []
        current: List[str] = []
        for token in input_tokens:
            if token != "[MASK]" and current:
                units.append(current)
                current = []
            current.append(token)
        if current:
            units.append(current)
        return units

    def _sliding_window(
        self,
        input_tokens: List[str],
        step: int = WINDOW_STEP,
        max_tokens: int = WINDOW_MAX_TOKENS,
    ) -> Tuple[List[int], List[int]]:
        """Ad-dabit(10): slide by word-units, aggregate with majority vote."""
        word_units = self._group_word_units(input_tokens)
        votes: Dict[int, List[int]] = defaultdict(list)

        for start in range(0, len(word_units), step):
            window_units: List[List[str]] = []
            token_count = 0
            for unit in word_units[start:]:
                unit_len = len(unit) + 2  # rough buffer for sub-word splits
                if token_count + unit_len > max_tokens:
                    break
                window_units.append(unit)
                token_count += unit_len

            if not window_units:
                break

            window_tokens = [t for unit in window_units for t in unit]
            preds, word_ids = self._forward(window_tokens)
            offset = sum(len(u) for u in word_units[:start])
            for p, w_id in zip(preds, word_ids):
                if w_id is not None:
                    votes[offset + w_id].append(p)

        total = len(input_tokens)
        final_preds = [
            Counter(votes[i]).most_common(1)[0][0] if votes[i] else -100
            for i in range(total)
        ]
        return final_preds, list(range(total))

    def _run_model(
        self, text: str
    ) -> Tuple[List[int], List[int], List[Tuple[str, int]], List[int | None]]:
        input_tokens, truth_tokens, word_info = self._tokenize(text)

        probe = self.tokenizer(input_tokens, is_split_into_words=True)
        if len(probe["input_ids"]) > MODEL_MAX_TOKENS:
            logger.debug("Sequence exceeds %d tokens, using sliding window", MODEL_MAX_TOKENS)
            predictions, word_ids = self._sliding_window(input_tokens)
        else:
            predictions, word_ids = self._forward(input_tokens)

        return truth_tokens, predictions, word_info, word_ids

    # -- Stage 3: reconstruction ----------------------------------------------

    def _reconstruct(
        self,
        predictions: List[int],
        word_info: List[Tuple[str, int]],
        word_ids: List[int | None],
    ) -> str:
        labels_aligned: List[str] = []
        last_word_id = None
        for i, w_id in enumerate(word_ids):
            if w_id is not None and w_id != last_word_id:
                labels_aligned.append(self.id2label[predictions[i]])
                last_word_id = w_id

        result: List[str] = []
        idx = 0
        for word, length in word_info:
            # labels_aligned[idx] is the label for the base word itself ('X').
            diacritics = labels_aligned[idx + 1 : idx + length + 1]
            rebuilt = ""
            for char, label in zip(word, diacritics):
                if (label == "تطويل"): 
                    label = "X" #Versi2: ketika label tatwil, dianggap tanpa diakritik
                harakat = HARAKAT_MAPPING.get(label, "")
                rebuilt += char + harakat
                
            result.append(rebuilt)
            idx += length + 1

        return " ".join(result)

    # -- Public --------------------------------------------------------------
    def filter_arabic_only(self, text):
        # Pattern untuk mencakup seluruh blok karakter Arab dan diakritiknya
        result = re.sub(r'[^\u0600-\u06FF\s]', '', text)

        # Membersihkan spasi ganda yang mungkin muncul setelah penghapusan
        return ' '.join(result.split())

    def diacritize(self, text: str) -> str:
        """Diacritize ``text`` and return the harakat-annotated string."""
        if not text or not text.strip():
            return ""
        _, predictions, word_info, word_ids = self._run_model(text)
        return self._reconstruct(predictions, word_info, word_ids)


# --- Module-level API --------------------------------------------------------

@lru_cache(maxsize=1)
def get_pipeline() -> AdDabitDiacritizationPipeline:
    """Singleton pipeline instance (safe to use as a FastAPI dependency)."""
    return AdDabitDiacritizationPipeline()


def diacritic_text(text: str) -> str:
    """Diacritize Arabic text — the public entry point used by the API."""
    return get_pipeline().diacritize(text)
