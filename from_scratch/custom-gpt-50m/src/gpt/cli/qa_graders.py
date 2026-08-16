"""Objectively gradeable prompts — the half of the QA set a machine can mark.

`gpt-eval` measures whether output is *well formed* (non-empty, non-repetitive, ASCII,
no role leakage). Those metrics saturate: once a model stops emitting garbage the score
pins near 100 and stops telling you anything. `test_loss` moves smoothly but is abstract
— it cannot tell you the model started getting arithmetic right.

This module fills the gap with three families of check that have a *right answer*:

1. **Fact / arithmetic** — the answer must contain a specific string or number.
2. **Constraint compliance** — the prompt imposed a mechanical rule ("exactly one word",
   "under 20 words", "do not mention France") that can be verified without judgement.
3. **Multiple choice by log-likelihood** — no generation at all. Score each candidate
   continuation's average token log-probability and take the argmax. This is how small
   base models are actually compared (lm-evaluation-harness works this way), and it is
   *deterministic*: no temperature, no seed, no sampling luck. Chance level is stated
   per item so a score is interpretable rather than just a number.

Deliberately not attempted: grading open-ended answers ("give two tips for sleeping").
A keyword rule there measures vocabulary, not quality, and would reward the model for
saying a magic word. Those stay in the human-read HTML report.
"""

import re

# ── graders ──────────────────────────────────────────────────────────────────


def _norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_any(*needles):
    """Passes if any needle appears (case-insensitive)."""
    def check(answer):
        a = _norm(answer)
        hit = next((n for n in needles if n.lower() in a), None)
        return hit is not None, f"found {hit!r}" if hit else f"none of {list(needles)}"
    return check


def final_number_is(expected, tolerance=1e-6):
    """Passes if the LAST number in the answer equals `expected`.

    The last number, not any number: a worked solution restates the inputs, so
    'contains 75' would pass on a wrong answer that merely echoed the question. The
    final figure is the model's actual claim.
    """
    def check(answer):
        nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer.replace("$", ""))
        if not nums:
            return False, "no number in answer"
        got = float(nums[-1].replace(",", ""))
        return abs(got - expected) <= tolerance, f"final number {got:g}, expected {expected:g}"
    return check


def word_count_at_most(n):
    def check(answer):
        c = len(answer.split())
        return c <= n, f"{c} words (limit {n})"
    return check


def exactly_one_word():
    def check(answer):
        words = [w for w in re.findall(r"[A-Za-z']+", answer)]
        return len(words) == 1, f"{len(words)} words"
    return check


def is_all_lowercase():
    def check(answer):
        letters = [c for c in answer if c.isalpha()]
        if not letters:
            return False, "no letters"
        bad = sum(1 for c in letters if c.isupper())
        return bad == 0, f"{bad} uppercase letters"
    return check


def does_not_contain(*needles):
    def check(answer):
        a = _norm(answer)
        hit = next((n for n in needles if n.lower() in a), None)
        return hit is None, f"mentioned {hit!r}" if hit else "clean"
    return check


def has_numbered_items(n):
    def check(answer):
        items = re.findall(r"^\s*\d+[.)]\s+\S", answer, re.M)
        return len(items) == n, f"{len(items)} numbered items (wanted {n})"
    return check


def avoids_letter(letter):
    def check(answer):
        c = answer.lower().count(letter.lower())
        return c == 0, f"{c} occurrences of {letter!r}"
    return check


# ── graded prompts: (category, prompt, grader) ───────────────────────────────
# Prompts here are duplicated from qa_prompts.py on purpose — the report asks all of
# them, this file marks only the subset with a defensible right answer.

GRADED_PROMPTS = [
    # Closed-book facts. Accepts synonyms where a correct answer has more than one form.
    ("facts", "What is the capital of France?", contains_any("paris")),
    ("facts", "What is the capital of Japan?", contains_any("tokyo")),
    ("facts", "Which planet in our solar system is known as the Red Planet?", contains_any("mars")),
    ("facts", "Who wrote the play Romeo and Juliet?", contains_any("shakespeare")),
    ("facts", "What is the chemical symbol for gold?", contains_any("au")),
    ("facts", "What is the largest ocean on Earth?", contains_any("pacific")),
    ("facts", "In what year did World War II end?", contains_any("1945")),
    ("facts", "What year did the first man land on the moon?", contains_any("1969")),
    ("facts", "What is the tallest mountain in the world?", contains_any("everest")),

    # Arithmetic — the gap GSM8K was added to close, now measured instead of eyeballed.
    ("arithmetic", "A bakery sold 48 cupcakes in the morning and 27 in the afternoon. "
                   "How many cupcakes did they sell in total?", final_number_is(75)),
    ("arithmetic", "Maria has $85. She spends $32 on groceries and $18 on a book. "
                   "How much money does she have left?", final_number_is(35)),
    ("arithmetic", "A school bus holds 36 students. If 8 buses are full, how many "
                   "students are being transported?", final_number_is(288)),
    ("arithmetic", "Tom read 24 pages of a book each day for 5 days. How many pages "
                   "did he read in total?", final_number_is(120)),
    ("arithmetic", "A pizza is cut into 8 equal slices. If 3 people each eat 2 slices, "
                   "how many slices are left?", final_number_is(2)),
    ("arithmetic", "A store gives a 20% discount on a $50 jacket. What is the final "
                   "price after the discount?", final_number_is(40)),
    ("arithmetic", "A tank holds 120 litres. It is 3/4 full. How many litres are in "
                   "the tank?", final_number_is(90)),
    ("arithmetic", "Sam earns $15 an hour and works 7 hours a day for 4 days. How much "
                   "does he earn?", final_number_is(420)),

    # Reasoning items that happen to have a single checkable answer.
    ("reasoning", "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?",
     contains_any("carol")),
    ("reasoning", "If it takes 5 machines 5 minutes to make 5 widgets, how long would "
                  "100 machines take to make 100 widgets?", contains_any("5 minutes", "five minutes")),
    ("reasoning", "You are in a race and you overtake the person in second place. "
                  "What position are you in now?", contains_any("second")),
    ("reasoning", "Yesterday was Tuesday. What day will it be the day after tomorrow?",
     contains_any("friday")),
    ("reasoning", "Which is heavier: a kilogram of feathers or a kilogram of iron?",
     contains_any("same", "equal", "neither")),

    # Constraint compliance — purely mechanical, no judgement required.
    ("constraints", "Answer in exactly one word: what colour is the sky on a clear day?",
     exactly_one_word()),
    ("constraints", "List exactly three fruits, as a numbered list, and nothing else.",
     has_numbered_items(3)),
    ("constraints", "Explain gravity in under 20 words.", word_count_at_most(20)),
    ("constraints", "Reply using only lowercase letters: WHAT IS YOUR NAME?", is_all_lowercase()),
    ("constraints", "Name a country in Europe. Do not mention France.", does_not_contain("france")),
    ("constraints", "Write a sentence about the ocean that does not contain the letter 'e'.",
     avoids_letter("e")),
]


# ── multiple choice, scored by log-likelihood (no generation) ────────────────
# (context, [choices], index of correct choice). Chance = 1/len(choices).
#
# Hand-written rather than downloaded: this runs offline and needs no dataset
# dependency. The trade is that scores are NOT comparable to published HellaSwag/ARC
# numbers — they are comparable to *this model at another checkpoint*, which is the
# question being asked. For published-comparable numbers use lm-evaluation-harness.

MC_ITEMS = [
    ("The capital city of France is", [" Paris", " Berlin", " Madrid", " Rome"], 0),
    ("Water freezes at a temperature of", [" 0 degrees Celsius", " 50 degrees Celsius",
                                           " 100 degrees Celsius", " 200 degrees Celsius"], 0),
    ("She put the ice cream in the freezer so it would", [" stay frozen", " melt quickly",
                                                          " catch fire", " grow taller"], 0),
    ("He was hungry, so he", [" made a sandwich", " read the ceiling",
                              " painted the rain", " sold his ears"], 0),
    ("The opposite of 'hot' is", [" cold", " warm", " loud", " green"], 0),
    ("A dog is a kind of", [" animal", " vegetable", " mineral", " building"], 0),
    ("To open a locked door you need a", [" key", " spoon", " cloud", " song"], 0),
    ("If it is raining outside you should take an", [" umbrella", " oven", " anchor", " piano"], 0),
    ("The sun rises in the", [" east", " west", " north", " south"], 0),
    ("Two plus two equals", [" four", " five", " seven", " ten"], 0),
    ("Birds are able to", [" fly", " photosynthesize", " breathe underwater", " melt"], 0),
    ("She studied hard for the exam because she wanted to",
     [" pass it", " fail it", " forget it", " eat it"], 0),
]
