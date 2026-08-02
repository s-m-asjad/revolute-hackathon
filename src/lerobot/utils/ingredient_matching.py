# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Detects which fillable ingredients (e.g. cucumber, tomato, cheese, lettuce -- as opposed to the bread,
which is handled separately) a free-text sandwich request explicitly names, and in what order.

This is deliberately a plain text match rather than an LLM call: the whole point is to preserve the exact
order the user said the ingredients in (e.g. "with tomato then cheese" vs "with cheese then tomato"), which
a paraphrase-tolerant LLM classifier would be free to reorder or drop.
"""

import re

# A few common irregular plurals/spellings for the known ingredients. Regular "+s" plurals (e.g.
# "cheeses") are handled automatically and don't need an entry here.
_INGREDIENT_SYNONYMS: dict[str, list[str]] = {
    "tomato": ["tomatoes"],
    "potato": ["potatoes"],
}


def _variant_pattern(name: str) -> re.Pattern:
    variants = {name, name if name.endswith("s") else f"{name}s"}
    variants.update(_INGREDIENT_SYNONYMS.get(name.lower(), []))
    alternation = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


def extract_requested_ingredients(prompt: str, known_ingredients: list[str]) -> list[str] | None:
    """Finds which of `known_ingredients` are explicitly named in `prompt`, in the order they're
    mentioned (first occurrence wins for each ingredient).

    Returns `None` if none of `known_ingredients` are mentioned at all -- meaning the request is generic
    (e.g. plain "make a sandwich") rather than naming specific fillings. Returns an (possibly empty) list
    of ingredient names (taken verbatim from `known_ingredients`, not from the prompt's wording) otherwise.
    """
    hits: list[tuple[int, str]] = []
    for name in known_ingredients:
        match = _variant_pattern(name).search(prompt)
        if match:
            hits.append((match.start(), name))

    if not hits:
        return None

    hits.sort(key=lambda hit: hit[0])
    return [name for _, name in hits]
