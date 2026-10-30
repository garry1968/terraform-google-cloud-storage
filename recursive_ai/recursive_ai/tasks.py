"""Generated reasoning tasks with known answers, for baseline evaluations.

Each task is built from a seed, so there are as many fresh tasks as needed
and none of them can be in a model's training data. Every possible task
belongs to exactly one split ("dev" or "holdout"), chosen by a hash of its
text, so the two can never share a task. The learning loop never imports
this module: holdout tasks must stay out of everything the system learns
from.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import random
import re
from collections import deque
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any

from .evaluator import has_term

SPLITS = ("dev", "holdout")
ANSWER_LINE = ("End your reply with a final line of the form "
               "'Answer: <answer>'.")

Grid = list[list[int]]


@dataclass
class Task:
    id: str
    category: str
    prompt: str
    answer: Any  # int, str or Grid
    kind: str  # "number", "name", "text" or "grid"
    choices: list[str] = field(default_factory=list)  # for "name" answers


# Mathematics ---------------------------------------------------------------

def _math(rng: random.Random, task_id: str) -> Task:
    form = rng.randrange(5)
    if form == 0:
        price, note = rng.randint(35, 95), rng.choice([5, 10, 20])
        count = rng.randint(3, 9)
        while count * price >= note * 100:
            count -= 1
        question = (f"Notebooks cost {price}p each. Sam buys {count} "
                    f"notebooks and pays with a £{note} note. How much change "
                    "does Sam get, in pence?")
        answer = note * 100 - count * price
    elif form == 1:
        a, b, c, d, e = (rng.randint(11, 99) for _ in range(5))
        question = f"What is ({a} * {b}) - ({c} * {d}) + {e}?"
        answer = a * b - c * d + e
    elif form == 2:
        s1, s2 = rng.randint(40, 120), rng.randint(40, 120)
        h1, h2 = rng.randint(2, 5), rng.randint(2, 5)
        question = (f"A train travels at {s1} km/h for {h1} hours, then at "
                    f"{s2} km/h for {h2} hours. How many kilometres does it "
                    "travel in total?")
        answer = s1 * h1 + s2 * h2
    elif form == 3:
        n, m = rng.randint(1000, 99999), rng.randint(7, 97)
        question = f"What is the remainder when {n} is divided by {m}?"
        answer = n % m
    else:
        width = rng.randint(3, 20)
        length = rng.randint(width + 1, 40)
        question = (f"A rectangle has a perimeter of {2 * (width + length)} "
                    f"cm and a width of {width} cm. What is its area in "
                    "square centimetres?")
        answer = width * length
    return Task(task_id, "math",
                f"{question} Answer with a whole number. {ANSWER_LINE}",
                answer, "number")


# Logic ---------------------------------------------------------------------

NAMES = ["Ava", "Ben", "Cara", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivan",
         "Jo"]
ORDINALS = ["1st", "2nd", "3rd", "4th", "5th"]


def _holds(clue: tuple, pos: dict[str, int]) -> bool:
    kind, x, *rest = clue
    if kind == "ahead":
        return pos[x] < pos[rest[0]]
    if kind == "behind":
        return pos[rest[0]] == pos[x] + 1
    if kind == "not_first":
        return pos[x] != 0
    return pos[x] != len(pos) - 1  # not_last


def _solutions(names: list[str], clues: list[tuple]) -> int:
    count = 0
    for perm in itertools.permutations(names):
        pos = {n: i for i, n in enumerate(perm)}
        count += all(_holds(c, pos) for c in clues)
    return count


def _logic(rng: random.Random, task_id: str) -> Task:
    names = rng.sample(NAMES, 5)
    order = rng.sample(names, 5)  # order[0] finished first
    pos = {n: i for i, n in enumerate(order)}
    pool = [("ahead", x, y) for x in names for y in names if pos[x] < pos[y]]
    pool += [("behind", x, y) for x in names for y in names
             if pos[y] == pos[x] + 1]
    pool += [("not_first", x) for x in names if pos[x] != 0]
    pool += [("not_last", x) for x in names if pos[x] != 4]
    rng.shuffle(pool)
    clues: list[tuple] = []
    for clue in pool:
        if _solutions(names, clues) == 1:
            break
        clues.append(clue)
    # Drop clues the answer does not need, so the puzzle stays short.
    for clue in list(clues):
        rest = [c for c in clues if c != clue]
        if _solutions(names, rest) == 1:
            clues = rest
    text = {"ahead": "{0} finished ahead of {1}.",
            "behind": "{1} finished directly behind {0}.",
            "not_first": "{0} did not finish first.",
            "not_last": "{0} did not finish last."}
    lines = "\n".join("- " + text[c[0]].format(*c[1:]) for c in clues)
    place = rng.randrange(5)
    prompt = (f"Five runners, {', '.join(sorted(names))}, finished a race in "
              f"different positions with no ties.\nClues:\n{lines}\n\nWho "
              f"finished {ORDINALS[place]}? {ANSWER_LINE}")
    return Task(task_id, "logic", prompt, order[place], "name", names)


# Unfamiliar rules: ARC-style grid puzzles ----------------------------------

def _flip_h(g: Grid) -> Grid:
    return [row[::-1] for row in g]


def _flip_v(g: Grid) -> Grid:
    return [list(row) for row in g[::-1]]


def _rot90(g: Grid) -> Grid:
    return [list(row) for row in zip(*g[::-1])]


def _rot180(g: Grid) -> Grid:
    return _flip_v(_flip_h(g))


def _transpose(g: Grid) -> Grid:
    return [list(row) for row in zip(*g)]


def _gravity(g: Grid) -> Grid:
    columns = []
    for col in zip(*g):
        filled = [v for v in col if v]
        columns.append([0] * (len(col) - len(filled)) + filled)
    return [list(row) for row in zip(*columns)]


FIXED_RULES: dict[str, Callable[[Grid], Grid]] = {
    "flip_h": _flip_h, "flip_v": _flip_v, "rot90": _rot90,
    "rot180": _rot180, "transpose": _transpose, "gravity": _gravity,
}


def _random_rule(rng: random.Random,
                 palette: list[int]) -> Callable[[Grid], Grid]:
    choice = rng.randrange(len(FIXED_RULES) + 3)
    if choice < len(FIXED_RULES):
        return list(FIXED_RULES.values())[choice]
    if choice == len(FIXED_RULES):
        shuffled = rng.sample(palette, len(palette))
        mapping = dict(zip(palette, shuffled))
        if all(k == v for k, v in mapping.items()):
            mapping = dict(zip(palette, palette[1:] + palette[:1]))
        return lambda g: [[mapping.get(v, v) for v in row] for row in g]
    if choice == len(FIXED_RULES) + 1:
        shift = rng.randint(1, 2)
        return lambda g: [row[-shift:] + row[:-shift] for row in g]
    colour = rng.choice([c for c in range(1, 10) if c not in palette])

    def border(g: Grid) -> Grid:
        h, w = len(g), len(g[0])
        return [[colour if r in (0, h - 1) or c in (0, w - 1) else g[r][c]
                 for c in range(w)] for r in range(h)]
    return border


def _render(g: Grid) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in g)


def _grid(rng: random.Random, task_id: str) -> Task:
    for _ in range(500):
        h, w = rng.randint(3, 6), rng.randint(3, 6)
        palette = rng.sample(range(1, 10), rng.randint(2, 3))
        steps = [_random_rule(rng, palette)
                 for _ in range(1 if rng.random() < 0.6 else 2)]

        def rule(g: Grid) -> Grid:
            for step in steps:
                g = step(g)
            return g

        # Every one- or two-step rule built from the fixed rules and this
        # puzzle's own recolour, shift or border steps.
        parts = list(FIXED_RULES.values()) + [
            s for s in steps if s not in FIXED_RULES.values()]
        alternatives = parts + [
            (lambda a, b: lambda g: b(a(g)))(a, b)
            for a, b in itertools.product(parts, repeat=2)]

        grids = [[[rng.choice(palette) if rng.random() < 0.35 else 0
                   for _ in range(w)] for _ in range(h)] for _ in range(4)]
        train, test = grids[:3], grids[3]
        outputs = [rule(g) for g in train]
        answer = rule(test)
        seen = {v for g in train for row in g for v in row}
        if (any(o == g for o, g in zip(outputs, train)) or answer == test
                or not {v for row in test for v in row} <= seen):
            continue
        # Reject a puzzle a different simple rule also explains.
        if any(all(alt(g) == o for g, o in zip(train, outputs))
               and alt(test) != answer for alt in alternatives):
            continue
        examples = "\n\n".join(
            f"Example {i}\nInput:\n{_render(g)}\nOutput:\n{_render(o)}"
            for i, (g, o) in enumerate(zip(train, outputs), 1))
        prompt = ("Each example shows an input grid and the output grid the "
                  "same hidden rule makes from it. Grids are rows of digits, "
                  f"and 0 is empty.\n\n{examples}\n\nTest input:\n"
                  f"{_render(test)}\n\nWhat is the output grid for the test "
                  "input? End your reply with a line 'Answer:' followed by "
                  "the output grid, one row per line.")
        return Task(task_id, "grid", prompt, answer, "grid")
    raise RuntimeError("could not generate an unambiguous grid puzzle")


# Planning: shortest route through a maze -----------------------------------

def _shortest(maze: list[list[str]], start: tuple[int, int],
              goal: tuple[int, int]) -> int | None:
    seen, queue = {start}, deque([(start, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == goal:
            return dist
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if (0 <= nr < len(maze) and 0 <= nc < len(maze[0])
                    and maze[nr][nc] != "#" and (nr, nc) not in seen):
                seen.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return None


def _planning(rng: random.Random, task_id: str) -> Task:
    while True:
        h, w = rng.randint(5, 7), rng.randint(5, 8)
        maze = [["#" if rng.random() < 0.3 else "." for _ in range(w)]
                for _ in range(h)]
        cells = [(r, c) for r in range(h) for c in range(w)
                 if maze[r][c] == "."]
        if len(cells) < 2:
            continue
        start, goal = rng.sample(cells, 2)
        dist = _shortest(maze, start, goal)
        if dist is not None and dist >= 6:
            break
    maze[start[0]][start[1]], maze[goal[0]][goal[1]] = "S", "G"
    picture = "\n".join("".join(row) for row in maze)
    prompt = ("In this maze, '#' is a wall and '.' is open floor. You start "
              "on S and want to reach G. Each move goes one square up, "
              "down, left or right, never into a wall or off the grid.\n\n"
              f"{picture}\n\nWhat is the smallest number of moves needed to "
              f"reach G? {ANSWER_LINE}")
    return Task(task_id, "planning", prompt, dist, "number")


# Coding: predict what a program prints -------------------------------------

WORDS = ["planet", "orange", "quartz", "banana", "eclipse", "rhythm",
         "violin", "cascade", "harbour", "minute", "garden", "silver",
         "lantern", "pyramid", "avocado", "biscuit", "dolphin", "festival",
         "giraffe", "horizon", "igloo", "journey", "kettle", "lemonade",
         "meadow", "notebook", "octopus", "pelican", "quiver", "sequoia"]


def _code(rng: random.Random, task_id: str) -> Task:
    form = rng.randrange(4)
    if form == 0:
        start, stop = rng.randint(1, 9), rng.randint(20, 60)
        step, factor = rng.randint(2, 7), rng.randint(2, 5)
        source = (f"total = 0\nfor i in range({start}, {stop}):\n"
                  f"    if i % {step} == 0:\n        total += i * {factor}\n"
                  "print(total)")
    elif form == 1:
        source = (f'word = "{rng.choice(WORDS)}"\nresult = ""\n'
                  'for ch in word:\n    if ch in "aeiou":\n'
                  "        result = ch + result\n    else:\n"
                  "        result = result + ch.upper()\nprint(result)")
    elif form == 2:
        numbers = [rng.randint(1, 99) for _ in range(7)]
        source = (f"xs = {numbers}\nys = sorted(x % {rng.randint(3, 9)} "
                  "for x in xs)\nprint(sum(ys[::2]) - ys[-1])")
    else:
        first, second = rng.randint(0, 5), rng.randint(0, 5)
        source = (f"def f(n):\n    if n == 0:\n        return {first}\n"
                  f"    if n == 1:\n        return {second}\n"
                  f"    return f(n - 1) + {rng.randint(1, 3)} * f(n - 2)"
                  f" - {rng.randint(0, 4)}\n\n"
                  f"print(f({rng.randint(5, 9)}))")
    printed = io.StringIO()
    with redirect_stdout(printed):
        exec(source, {})  # noqa: S102 - our own generated program
    output = printed.getvalue().strip()
    is_number = output.lstrip("-").isdigit()
    prompt = (f"What does this Python program print?\n\n```python\n{source}\n"
              f"```\n\n{ANSWER_LINE}")
    return Task(task_id, "code", prompt, int(output) if is_number else output,
                "number" if is_number else "text")


# Adaptation: discover a hidden rule from examples ---------------------------

def _digit_sum(x: int) -> int:
    return sum(int(d) for d in str(x))


RULE_FAMILIES: dict[str, Callable[[int, int, int], int]] = {
    "linear": lambda x, a, b: a * x + b,
    "square": lambda x, a, b: x * x + a * x + b,
    "digit_sum": lambda x, a, b: _digit_sum(x) * a + b,
    "modulo": lambda x, a, b: x % a + b,
    "reverse": lambda x, a, b: int(str(x)[::-1]) * a + b,
    "halve_or_grow": lambda x, a, b: x // 2 + b if x % 2 == 0 else a * x + b,
}
A_RANGE, B_RANGE = range(2, 10), range(-10, 21)


def _adaptive(rng: random.Random, task_id: str) -> Task:
    for _ in range(500):
        family = rng.choice(list(RULE_FAMILIES.values()))
        a, b = rng.choice(A_RANGE), rng.choice(B_RANGE)
        xs = rng.sample(range(10, 100), 6)
        examples, query = xs[:5], xs[5]
        outputs = [family(x, a, b) for x in examples]
        answer = family(query, a, b)
        # Reject a rule another family or setting also explains.
        if any(all(f(x, p, q) == y for x, y in zip(examples, outputs))
               and f(query, p, q) != answer
               for f in RULE_FAMILIES.values()
               for p in A_RANGE for q in B_RANGE):
            continue
        lines = "\n".join(f"{x} -> {y}" for x, y in zip(examples, outputs))
        prompt = ("A hidden rule turns each input number into an output "
                  f"number:\n{lines}\n\nWhat does the rule give for {query}? "
                  f"Answer with a whole number. {ANSWER_LINE}")
        return Task(task_id, "adaptive", prompt, answer, "number")
    raise RuntimeError("could not generate an unambiguous rule")


GENERATORS: dict[str, Callable[[random.Random, str], Task]] = {
    "math": _math, "logic": _logic, "grid": _grid, "planning": _planning,
    "code": _code, "adaptive": _adaptive,
}
CATEGORIES = tuple(GENERATORS)


def split_of(prompt: str) -> str:
    return SPLITS[hashlib.sha256(prompt.encode()).digest()[0] % 2]


def make_tasks(split: str, per_category: int,
               categories: tuple[str, ...] = CATEGORIES) -> list[Task]:
    """The same split and size always give the same tasks, with no repeats.
    Categories are interleaved, so a run that stops early still covers all
    of them."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    tasks, seen = [], set()
    for i in range(per_category):
        for cat in categories:
            rng = random.Random(f"{split}/{cat}/{i}")
            while True:
                task = GENERATORS[cat](rng, f"{split}-{cat}-{i}")
                if split_of(task.prompt) == split \
                        and task.prompt not in seen:
                    break
            seen.add(task.prompt)
            tasks.append(task)
    return tasks


# Grading -------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_answer(reply: str) -> str:
    """The text after the last "Answer:", or the whole reply if none."""
    at = reply.lower().rfind("answer:")
    return reply[at + len("answer:"):] if at >= 0 else reply


def parse_grid(text: str) -> Grid:
    """The last block of digit-only lines, e.g. "1 0 2" or "102"."""
    blocks: list[Grid] = [[]]
    for line in text.splitlines():
        cleaned = re.sub(r"[`|,\[\]]", " ", line).strip()
        if cleaned and re.fullmatch(r"[\d ]+", cleaned):
            blocks[-1].append([int(ch) for ch in cleaned if ch.isdigit()])
        elif blocks[-1]:
            blocks.append([])
    return next((b for b in reversed(blocks) if b), [])


def grade(task: Task, reply: str) -> bool:
    final = final_answer(reply).strip()
    if task.kind == "number":
        numbers = _NUMBER.findall(final)
        if not numbers:
            return False
        # The first number after "Answer:"; without one, the last number.
        value = numbers[0] if "answer:" in reply.lower() else numbers[-1]
        return float(value.replace(",", "")) == float(task.answer)
    if task.kind == "name":
        return [n for n in task.choices if has_term(final, n)] == [task.answer]
    if task.kind == "text":
        first = final.splitlines()[0] if final else ""
        return first.strip().strip("`'\"*") == task.answer
    return parse_grid(final) == task.answer
