"""Generated reasoning tasks with known answers, for baseline evaluations.

Tasks come in families. A family is one kind of problem, such as sliding-tile
puzzles, and belongs to exactly one split. So the "holdout" split holds kinds
of problem that never appear in "dev", not just new instances of them.
Answers come from code (search, brute force, or running our own programs),
never from a model. The learning loop never imports this module.
"""

from __future__ import annotations

import io
import itertools
import random
import re
from collections import deque
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from .evaluator import has_term

SPLITS = ("dev", "holdout")
CATEGORIES = ("math", "logic", "grid", "planning", "code", "rules")
ANSWER_LINE = ("End your reply with a final line of the form "
               "'Answer: <answer>'.")
WHOLE_NUMBER = f" Answer with a whole number. {ANSWER_LINE}"

Grid = list[list[int]]
# What a family generator returns: prompt, answer, kind and choices.
Generated = tuple[str, Any, str, list[str]]


@dataclass
class Task:
    id: str
    category: str
    family: str
    prompt: str
    answer: Any  # int, str, list of names, or Grid
    kind: str  # "number", "name", "names", "text" or "grid"
    choices: list[str] = field(default_factory=list)  # for name answers


def _run(source: str) -> str:
    """Run one of our own generated programs and return what it prints."""
    printed = io.StringIO()
    with redirect_stdout(printed):
        exec(source, {})  # noqa: S102 - generated here, never by a model
    return printed.getvalue().strip()


# Mathematics ---------------------------------------------------------------

def _divisible(rng: random.Random) -> Generated:
    n = rng.randint(800, 9000)
    a, b = rng.sample(range(3, 14), 2)
    c = rng.choice([x for x in range(4, 40) if x not in (a, b)])
    count = sum(1 for k in range(1, n + 1)
                if (k % a == 0 or k % b == 0) and k % c)
    return (f"How many whole numbers from 1 to {n} are divisible by {a} or "
            f"by {b}, but not by {c}?{WHOLE_NUMBER}", count, "number", [])


def _digit_sums(rng: random.Random) -> Generated:
    digits = rng.choice([4, 5])
    total = rng.randint(digits + 3, 9 * digits - 6)
    count = sum(1 for k in range(10 ** (digits - 1), 10 ** digits)
                if sum(map(int, str(k))) == total)
    return (f"How many {digits}-digit whole numbers have digits that add up "
            f"to exactly {total}?{WHOLE_NUMBER}", count, "number", [])


def _coin_ways(coins: list[int], amount: int) -> int:
    ways = [1] + [0] * amount
    for coin in coins:
        for value in range(coin, amount + 1):
            ways[value] += ways[value - coin]
    return ways[amount]


def _coins(rng: random.Random) -> Generated:
    while True:
        coins = sorted(rng.sample([1, 2, 5, 10, 20, 50], rng.randint(3, 4)))
        amount = rng.randint(40, 150)
        ways = _coin_ways(coins, amount)
        if ways >= 5:
            break
    names = ", ".join(f"{c}p" for c in coins)
    return (f"You have unlimited coins worth {names}. In how many different "
            f"ways can you make exactly {amount}p? The order of the coins "
            f"does not matter.{WHOLE_NUMBER}", ways, "number", [])


def _powers(rng: random.Random) -> Generated:
    base, power = rng.randint(3, 99), rng.randint(40, 999)
    modulus = rng.choice([7, 13, 97, 100, 360, 1000])
    return (f"What is the remainder when {base} to the power {power} is "
            f"divided by {modulus}?{WHOLE_NUMBER}",
            pow(base, power, modulus), "number", [])


# Logic ---------------------------------------------------------------------

NAMES = ["Ava", "Ben", "Cara", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivan",
         "Jo"]
ORDINALS = ["1st", "2nd", "3rd", "4th", "5th", "6th"]


def _minimal_clues(rng: random.Random, pool: list, worlds: list,
                   holds: Callable[[Any, Any], bool]) -> list:
    """True clues that leave exactly one world, with none to spare."""
    rng.shuffle(pool)
    clues, left = [], worlds
    for clue in pool:
        if len(left) == 1:
            break
        narrowed = [w for w in left if holds(clue, w)]
        if len(narrowed) < len(left):
            clues.append(clue)
            left = narrowed
    for clue in list(clues):
        rest = [c for c in clues if c != clue]
        if sum(all(holds(c, w) for c in rest) for w in worlds) == 1:
            clues = rest
    return clues


def _race_holds(clue: tuple, pos: dict[str, int]) -> bool:
    kind, *args = clue
    if kind == "ahead":
        return pos[args[0]] < pos[args[1]]
    if kind == "gap":
        return abs(pos[args[0]] - pos[args[1]]) - 1 == args[2]
    if kind == "apart":
        return abs(pos[args[0]] - pos[args[1]]) > 1
    return pos[args[0]] % 2 == (1 if kind == "even" else 0)


def _race(rng: random.Random) -> Generated:
    names = rng.sample(NAMES, 6)
    order = rng.sample(names, 6)
    pos = {n: i for i, n in enumerate(order)}
    pairs = list(itertools.combinations(names, 2))
    pool = [("ahead", x, y) for x, y in itertools.permutations(names, 2)
            if pos[x] < pos[y]]
    pool += [("gap", x, y, abs(pos[x] - pos[y]) - 1) for x, y in pairs
             if abs(pos[x] - pos[y]) <= 3]
    pool += [("apart", x, y) for x, y in pairs if abs(pos[x] - pos[y]) > 1]
    pool += [("even" if pos[x] % 2 else "odd", x) for x in names]
    worlds = [{n: i for i, n in enumerate(p)}
              for p in itertools.permutations(names)]
    clues = _minimal_clues(rng, pool, worlds, _race_holds)

    def say(clue: tuple) -> str:
        kind, *args = clue
        if kind == "ahead":
            return f"{args[0]} finished ahead of {args[1]}."
        if kind == "gap" and args[2] == 0:
            return f"{args[0]} and {args[1]} finished next to each other."
        if kind == "gap":
            runners = "runner" if args[2] == 1 else "runners"
            return (f"Exactly {args[2]} {runners} finished between "
                    f"{args[0]} and {args[1]}.")
        if kind == "apart":
            return (f"{args[0]} and {args[1]} did not finish next to each "
                    "other.")
        return f"{args[0]} finished in an {kind}-numbered place."

    place = rng.randrange(6)
    lines = "\n".join(f"- {say(c)}" for c in clues)
    return (f"Six runners, {', '.join(sorted(names))}, finished a race in "
            f"different places with no ties.\nClues:\n{lines}\n\nWho "
            f"finished {ORDINALS[place]}? {ANSWER_LINE}", order[place],
            "name", names)


def _knight_statements(names: list[str], speaker: str) -> list[tuple]:
    others = [n for n in names if n != speaker]
    options: list[tuple] = []
    for y in others:
        options.append((f"{y} is a knave.", lambda w, y=y: not w[y]))
        options.append((f"{y} is a knight.", lambda w, y=y: w[y]))
    for k in range(1, len(names)):
        options.append((f"Exactly {k} of us are knights.",
                        lambda w, k=k: sum(w.values()) == k))
    for y, z in itertools.combinations(others, 2):
        options.append((f"{y} and {z} are the same kind.",
                        lambda w, y=y, z=z: w[y] == w[z]))
        options.append((f"At least one of {y} and {z} is a knave.",
                        lambda w, y=y, z=z: not (w[y] and w[z])))
    return options


def _knights(rng: random.Random) -> Generated:
    names = sorted(rng.sample(NAMES, 4))
    worlds = [dict(zip(names, bits))
              for bits in itertools.product([True, False], repeat=4)]
    for _ in range(1000):
        truth = rng.choice(worlds[1:-1])  # at least one of each kind
        said = {}
        for speaker in names:
            fitting = [s for s in _knight_statements(names, speaker)
                       if s[1](truth) == truth[speaker]]
            said[speaker] = rng.choice(fitting)
        consistent = [w for w in worlds
                      if all(said[s][1](w) == w[s] for s in names)]
        if len(consistent) == 1:
            break
    else:
        raise RuntimeError("could not generate a knights puzzle")
    lines = "\n".join(f'- {s} says: "{said[s][0]}"' for s in names)
    knights = [n for n in names if truth[n]]
    return ("On an island, knights always tell the truth and knaves always "
            "lie, and everyone is one or the other. Four islanders speak:\n"
            f"{lines}\n\nWhich of them are knights? Give all their names. "
            f"{ANSWER_LINE}", knights, "names", names)


PETS = ["cat", "dog", "fish", "parrot"]
DRINKS = ["tea", "coffee", "milk", "juice"]


def _pets_holds(clue: tuple, world: tuple[dict, dict]) -> bool:
    pet, drink = world
    kind, a, b = clue
    owner = {p: person for person, p in pet.items()}
    if kind == "pet_drink":
        return drink[owner[a]] == b
    if kind == "pet_not_drink":
        return drink[owner[a]] != b
    if kind == "not_pet":
        return pet[a] != b
    if kind == "drinks":
        return drink[a] == b
    return drink[a] != b  # not_drink


def _pets(rng: random.Random) -> Generated:
    people = sorted(rng.sample(NAMES, 4))
    world = (dict(zip(people, rng.sample(PETS, 4))),
             dict(zip(people, rng.sample(DRINKS, 4))))
    worlds = [(dict(zip(people, p)), dict(zip(people, d)))
              for p in itertools.permutations(PETS)
              for d in itertools.permutations(DRINKS)]
    candidates = (
        [("pet_drink", p, d) for p in PETS for d in DRINKS]
        + [("pet_not_drink", p, d) for p in PETS for d in DRINKS]
        + [("not_pet", person, p) for person in people for p in PETS]
        + [("drinks", person, d) for person in people for d in DRINKS]
        + [("not_drink", person, d) for person in people for d in DRINKS])
    pool = [c for c in candidates if _pets_holds(c, world)]
    clues = _minimal_clues(rng, pool, worlds, _pets_holds)
    text = {"pet_drink": "The person with the {0} drinks {1}.",
            "pet_not_drink": "The person with the {0} does not drink {1}.",
            "not_pet": "{0} does not have the {1}.",
            "drinks": "{0} drinks {1}.",
            "not_drink": "{0} does not drink {1}."}
    lines = "\n".join("- " + text[c[0]].format(*c[1:]) for c in clues)
    pet = rng.choice(PETS)
    owner = next(person for person, p in world[0].items() if p == pet)
    return (f"Four friends, {', '.join(people)}, each have a different pet "
            f"({', '.join(PETS)}) and each like a different drink "
            f"({', '.join(DRINKS)}).\nClues:\n{lines}\n\nWho has the {pet}? "
            f"{ANSWER_LINE}", owner, "name", people)


def _colourings(edges: list[tuple[str, str]], regions: str = "ABCDE") -> int:
    return sum(1 for colours in itertools.product(range(3),
                                                  repeat=len(regions))
               if all(colours[regions.index(a)] != colours[regions.index(b)]
                      for a, b in edges))


def _colouring(rng: random.Random) -> Generated:
    while True:
        edges = sorted(rng.sample(list(itertools.combinations("ABCDE", 2)),
                                  rng.randint(5, 8)))
        count = _colourings(edges)
        if count:
            break
    borders = ", ".join(f"{a}-{b}" for a, b in edges)
    return ("A map has five regions, A to E. These pairs of regions share a "
            f"border: {borders}. Each region is painted red, green or blue, "
            "and regions that share a border must be different colours. In "
            f"how many different ways can the map be painted?{WHOLE_NUMBER}",
            count, "number", [])


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


def _objects(g: Grid) -> list[list[tuple[int, int]]]:
    """Groups of touching (up, down, left, right) non-empty cells."""
    seen, groups = set(), []
    for r, row in enumerate(g):
        for c, v in enumerate(row):
            if v and (r, c) not in seen:
                group, todo = [], [(r, c)]
                seen.add((r, c))
                while todo:
                    cr, cc = todo.pop()
                    group.append((cr, cc))
                    for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1),
                                   (cr, cc - 1)):
                        if (0 <= nr < len(g) and 0 <= nc < len(row)
                                and g[nr][nc] and (nr, nc) not in seen):
                            seen.add((nr, nc))
                            todo.append((nr, nc))
                groups.append(group)
    return groups


def _keep_largest(g: Grid) -> Grid:
    groups = _objects(g)
    biggest = max((len(grp) for grp in groups), default=0)
    keep = {cell for grp in groups if len(grp) == biggest for cell in grp}
    return [[v if (r, c) in keep else 0 for c, v in enumerate(row)]
            for r, row in enumerate(g)]


def _drop_singles(g: Grid) -> Grid:
    single = {grp[0] for grp in _objects(g) if len(grp) == 1}
    return [[0 if (r, c) in single else v for c, v in enumerate(row)]
            for r, row in enumerate(g)]


def _fill_down(g: Grid) -> Grid:
    out = [list(row) for row in g]
    for c in range(len(g[0])):
        colour = 0
        for r in range(len(g)):
            colour = out[r][c] = out[r][c] or colour
    return out


def _fill_right(g: Grid) -> Grid:
    return _transpose(_fill_down(_transpose(g)))


def _fill_up(g: Grid) -> Grid:
    return _flip_v(_fill_down(_flip_v(g)))


def _fill_left(g: Grid) -> Grid:
    return _flip_h(_fill_right(_flip_h(g)))


def _mirror_lr(g: Grid) -> Grid:
    half = len(g[0]) // 2
    return [row[:len(row) - half] + row[:half][::-1] for row in g]


def _mirror_tb(g: Grid) -> Grid:
    return _transpose(_mirror_lr(_transpose(g)))


GEOMETRY = [_flip_h, _flip_v, _rot90, _rot180, _transpose, _gravity]
OBJECT_RULES = [_keep_largest, _drop_singles]
FILL_RULES = [_fill_down, _fill_right, _fill_up, _fill_left]
MIRROR_RULES = [_mirror_lr, _mirror_tb]
ALL_RULES = GEOMETRY + OBJECT_RULES + FILL_RULES + MIRROR_RULES


def _recolour(rng: random.Random,
              palette: list[int]) -> Callable[[Grid], Grid]:
    mapping = dict(zip(palette, palette[1:] + palette[:1]))
    if rng.random() < 0.5:
        mapping = dict(zip(palette, palette[::-1]))
        if len(palette) % 2:  # the middle colour would stay put
            mapping = dict(zip(palette, palette[1:] + palette[:1]))
    return lambda g: [[mapping.get(v, v) for v in row] for row in g]


def _grid_puzzle(rng: random.Random,
                 pick_steps: Callable[[list[int]], list]) -> Generated:
    for _ in range(500):
        h, w = rng.randint(5, 8), rng.randint(5, 8)
        palette = rng.sample(range(1, 10), rng.randint(2, 4))
        steps = pick_steps(palette)

        def rule(g: Grid, steps: list = steps) -> Grid:
            for step in steps:
                g = step(g)
            return g

        grids = [[[rng.choice(palette) if rng.random() < 0.3 else 0
                   for _ in range(w)] for _ in range(h)] for _ in range(4)]
        train, test = grids[:3], grids[3]
        outputs = [rule(g) for g in train]
        answer = rule(test)
        seen = {v for g in train for row in g for v in row}
        if (any(o == g for o, g in zip(outputs, train)) or answer == test
                or not {v for row in test for v in row} <= seen):
            continue
        # Reject a puzzle that another one- or two-step rule also explains.
        parts = ALL_RULES + [s for s in steps if s not in ALL_RULES]
        alternatives = parts + [(lambda a, b: lambda g: b(a(g)))(a, b)
                                for a, b in itertools.product(parts,
                                                              repeat=2)]
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
        return prompt, answer, "grid", []
    raise RuntimeError("could not generate an unambiguous grid puzzle")


def _render(g: Grid) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in g)


def _grid_geometry(rng: random.Random) -> Generated:
    return _grid_puzzle(rng, lambda palette: [
        rng.choice(GEOMETRY),
        rng.choice(GEOMETRY + [_recolour(rng, palette)])])


def _grid_objects(rng: random.Random) -> Generated:
    return _grid_puzzle(rng, lambda palette: [rng.choice(OBJECT_RULES)] + (
        [rng.choice(GEOMETRY)] if rng.random() < 0.5 else []))


def _grid_fill(rng: random.Random) -> Generated:
    return _grid_puzzle(rng, lambda palette: [rng.choice(FILL_RULES)] + (
        [_recolour(rng, palette)] if rng.random() < 0.4 else []))


def _grid_mirror(rng: random.Random) -> Generated:
    return _grid_puzzle(rng, lambda palette: [rng.choice(MIRROR_RULES)] + (
        [rng.choice(FILL_RULES)] if rng.random() < 0.4 else []))


# Planning: shortest solutions found by search ------------------------------

STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _key_maze_moves(maze: list[str], start: tuple[int, int]) -> int | None:
    """Fewest moves from start to G, where D opens only after visiting k."""
    first = (start, maze[start[0]][start[1]] == "k")
    seen, queue = {first}, deque([(first, 0)])
    while queue:
        ((r, c), key), dist = queue.popleft()
        if maze[r][c] == "G":
            return dist
        for dr, dc in STEPS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < len(maze) and 0 <= nc < len(maze[0])):
                continue
            cell = maze[nr][nc]
            if cell == "#" or (cell == "D" and not key):
                continue
            state = ((nr, nc), key or cell == "k")
            if state not in seen:
                seen.add(state)
                queue.append((state, dist + 1))
    return None


def _key_maze(rng: random.Random) -> Generated:
    while True:
        h, w = rng.randint(6, 8), rng.randint(7, 9)
        cells = [["#" if rng.random() < 0.28 else "." for _ in range(w)]
                 for _ in range(h)]
        open_cells = [(r, c) for r in range(h) for c in range(w)
                      if cells[r][c] == "."]
        if len(open_cells) < 4:
            continue
        start, goal, key, door = rng.sample(open_cells, 4)
        for (r, c), mark in ((goal, "G"), (key, "k"), (door, "D")):
            cells[r][c] = mark
        maze = ["".join(row) for row in cells]
        without_key = [row.replace("k", ".") for row in maze]
        moves = _key_maze_moves(maze, start)
        if (moves is not None and moves >= 10
                and _key_maze_moves(without_key, start) is None):
            break
    cells[start[0]][start[1]] = "S"
    picture = "\n".join("".join(row) for row in cells)
    return ("In this maze, '#' is a wall and '.' is open floor. 'k' is a key "
            "and 'D' is a locked door: you can step onto D only after you "
            "have picked up the key by stepping onto k. You start on S, and "
            "each move goes one square up, down, left or right.\n\n"
            f"{picture}\n\nWhat is the smallest number of moves needed to "
            f"reach G? {ANSWER_LINE}", moves, "number", [])


def _jug_steps(a: int, b: int, target: int) -> int | None:
    seen, queue = {(0, 0)}, deque([((0, 0), 0)])
    while queue:
        (x, y), dist = queue.popleft()
        if target in (x, y):
            return dist
        pour_xy, pour_yx = min(x, b - y), min(y, a - x)
        for state in ((a, y), (x, b), (0, y), (x, 0),
                      (x - pour_xy, y + pour_xy), (x + pour_yx, y - pour_yx)):
            if state not in seen:
                seen.add(state)
                queue.append((state, dist + 1))
    return None


def _jugs(rng: random.Random) -> Generated:
    while True:
        a, b = sorted(rng.sample(range(3, 16), 2))
        target = rng.randint(1, b - 1)
        steps = _jug_steps(a, b, target)
        if target != a and steps is not None and steps >= 5:
            break
    return (f"You have an empty {a}-litre jug, an empty {b}-litre jug and a "
            "tap. One step is filling a jug to the top from the tap, "
            "emptying a jug completely, or pouring from one jug into the "
            "other until the first is empty or the second is full. What is "
            f"the smallest number of steps needed to have exactly {target} "
            f"litres in one of the jugs?{WHOLE_NUMBER}", steps, "number", [])


def _ice_moves(floor: list[str], start: tuple[int, int]) -> int | None:
    seen, queue = {start}, deque([(start, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        if floor[r][c] == "G":
            return dist
        for dr, dc in STEPS:
            nr, nc = r, c
            while (0 <= nr + dr < len(floor) and 0 <= nc + dc < len(floor[0])
                   and floor[nr + dr][nc + dc] != "#"):
                nr, nc = nr + dr, nc + dc
            if (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return None


def _ice(rng: random.Random) -> Generated:
    while True:
        h, w = rng.randint(7, 9), rng.randint(7, 9)
        cells = [["#" if rng.random() < 0.18 else "." for _ in range(w)]
                 for _ in range(h)]
        open_cells = [(r, c) for r in range(h) for c in range(w)
                      if cells[r][c] == "."]
        if len(open_cells) < 2:
            continue
        start, goal = rng.sample(open_cells, 2)
        cells[goal[0]][goal[1]] = "G"
        moves = _ice_moves(["".join(row) for row in cells], start)
        if moves is not None and moves >= 4:
            break
    cells[start[0]][start[1]] = "S"
    picture = "\n".join("".join(row) for row in cells)
    return ("This floor is ice: '#' is a rock and '.' is ice. You start on "
            "S. Each move, you pick up, down, left or right and slide that "
            "way until the next square is a rock or the edge of the floor; "
            "you cannot stop part-way. You reach G only if a slide ends on "
            f"G.\n\n{picture}\n\nWhat is the smallest number of moves needed "
            f"to reach G? {ANSWER_LINE}", moves, "number", [])


SOLVED = (1, 2, 3, 4, 5, 6, 7, 8, 0)


@lru_cache(maxsize=1)
def _tiles_by_distance() -> dict[int, list[tuple[int, ...]]]:
    """Every solvable layout of the 3x3 sliding puzzle, by fewest moves."""
    dist, queue = {SOLVED: 0}, deque([SOLVED])
    while queue:
        state = queue.popleft()
        gap = state.index(0)
        r, c = divmod(gap, 3)
        for dr, dc in STEPS:
            if 0 <= r + dr < 3 and 0 <= c + dc < 3:
                swap = (r + dr) * 3 + c + dc
                cells = list(state)
                cells[gap], cells[swap] = cells[swap], cells[gap]
                after = tuple(cells)
                if after not in dist:
                    dist[after] = dist[state] + 1
                    queue.append(after)
    layouts: dict[int, list[tuple[int, ...]]] = {}
    for state, moves in dist.items():
        layouts.setdefault(moves, []).append(state)
    return layouts


def _board(state: tuple[int, ...]) -> str:
    return "\n".join(" ".join(str(v) if v else "_" for v in state[i:i + 3])
                     for i in (0, 3, 6))


def _tiles(rng: random.Random) -> Generated:
    moves = rng.randint(10, 18)
    state = rng.choice(_tiles_by_distance()[moves])
    return ("This is a sliding puzzle on a 3 by 3 board with eight numbered "
            "tiles and a gap (_). A move slides one tile that is next to the "
            "gap into the gap. Starting from this layout:\n\n"
            f"{_board(state)}\n\nwhat is the smallest number of moves needed "
            f"to reach this layout?\n\n{_board(SOLVED)}\n\n{ANSWER_LINE}",
            moves, "number", [])


# Coding: questions about programs, answered by running them ----------------

WORDS = ["planet", "orange", "quartz", "banana", "eclipse", "rhythm",
         "violin", "cascade", "harbour", "minute", "garden", "silver",
         "lantern", "pyramid", "avocado", "biscuit", "dolphin", "festival",
         "giraffe", "horizon", "journey", "kettle", "meadow", "octopus"]


def _trace(rng: random.Random) -> Generated:
    form = rng.randrange(3)
    if form == 0:
        a, b, k = rng.randint(4, 9), rng.randint(8, 14), rng.randint(3, 7)
        source = (f"total = 0\nfor i in range(1, {a}):\n"
                  f"    for j in range(i, {b}):\n"
                  f"        if (i * j) % {k} == {rng.randrange(k)}:\n"
                  "            total += i - j\nprint(total)")
    elif form == 1:
        source = (f'word = "{rng.choice(WORDS)}"\nout = []\n'
                  "for i, ch in enumerate(word):\n"
                  f"    if i % {rng.randint(2, 4)} == 0:\n"
                  "        out.append(ch.upper())\n"
                  '    elif ch in "aeiou":\n        out.insert(0, ch)\n'
                  "    else:\n        out.append(str(i))\n"
                  'print("".join(out))')
    else:
        numbers = [rng.randint(2, 30) for _ in range(rng.randint(5, 7))]
        source = (f"xs = {numbers}\nwhile len(xs) > 1:\n"
                  "    a = xs.pop(0)\n    b = xs.pop(0)\n"
                  f"    xs.append((a * b + {rng.randint(1, 9)}) % "
                  f"{rng.randint(11, 50)})\nprint(xs[0])")
    output = _run(source)
    is_number = output.lstrip("-").isdigit()
    return (f"What does this Python program print?\n\n```python\n{source}\n"
            f"```\n\n{ANSWER_LINE}", int(output) if is_number else output,
            "number" if is_number else "text", [])


def _inverse(rng: random.Random) -> Generated:
    while True:
        a, b = rng.randint(3, 19), rng.randint(4, 15)
        if rng.random() < 0.5:
            body = (f"    digits = sum(int(d) for d in str(n))\n"
                    f"    return n % {a} == {rng.randrange(a)} and "
                    f"digits == {rng.randint(10, 30)} and n % {b} != 0")
        else:
            body = (f'    return bin(n).count("1") == {rng.randint(3, 7)} '
                    f"and n % {a} == {rng.randrange(a)} and n % {b} != 0")
        source = f"def check(n):\n{body}"
        namespace: dict[str, Any] = {}
        exec(source, namespace)  # noqa: S102 - generated here
        answer = next((n for n in range(1, 200_000)
                       if namespace["check"](n)), None)
        if answer is not None and answer > 50:
            break
    return (f"Here is a Python function:\n\n```python\n{source}\n```\n\nWhat "
            "is the smallest positive whole number n for which check(n) "
            f"returns True?{WHOLE_NUMBER}", answer, "number", [])


def _recursion(rng: random.Random) -> Generated:
    a, b, c = rng.randint(1, 3), rng.randint(1, 4), rng.randint(0, 5)
    source = ("def f(n, k):\n    if n <= 1 or k == 0:\n        return n + k\n"
              "    if n % 2 == 0:\n"
              f"        return f(n - 1, k) + {a} * f(n // 2, k - 1)\n"
              f"    return f(n - 2, k) - {b} * k + {c}\n\n"
              f"print(f({rng.randint(6, 11)}, {rng.randint(2, 4)}))")
    return (f"What does this Python program print?\n\n```python\n{source}\n"
            f"```\n\n{ANSWER_LINE}", int(_run(source)), "number", [])


def _count_strings(rng: random.Random) -> Generated:
    while True:
        pattern = rng.choice(["aa", "bb", "aba", "bab", "abb", "bba"])
        length = rng.randint(8, 12)
        bs = rng.randint(2, length // 2)
        source = (f'def ok(s):\n    return "{pattern}" not in s and '
                  f's.count("b") == {bs}')
        namespace: dict[str, Any] = {}
        exec(source, namespace)  # noqa: S102 - generated here
        count = sum(1 for letters in itertools.product("ab", repeat=length)
                    if namespace["ok"]("".join(letters)))
        if count >= 2:
            break
    return (f"Here is a Python function:\n\n```python\n{source}\n```\n\nHow "
            f"many different strings of length {length}, made only of the "
            f"letters a and b, make ok return True?{WHOLE_NUMBER}", count,
            "number", [])


# Rules: discover a hidden rule from examples -------------------------------

def _digit_sum(x: int) -> int:
    return sum(int(d) for d in str(x))


RULE_TYPES: dict[str, Callable[[int, int, int], int]] = {
    "linear": lambda x, a, b: a * x + b,
    "quadratic": lambda x, a, b: x * x - a * x + b,
    "digit_sum": lambda x, a, b: _digit_sum(x) * a + b,
    "scaled_mod": lambda x, a, b: (x % a) * (b % 7 + 2),
    "ones": lambda x, a, b: bin(x).count("1") * a + b,
    "xor": lambda x, a, b: (x ^ a) + b,
    "reverse": lambda x, a, b: int(str(x)[::-1]) * a + b,
    "halve_or_grow": lambda x, a, b: x // 2 + b if x % 2 == 0 else a * x + b,
}
A_RANGE, B_RANGE = range(2, 13), range(-20, 31)


def _rule_puzzle(rng: random.Random, types: list[str]) -> Generated:
    for _ in range(500):
        rule = RULE_TYPES[rng.choice(types)]
        a, b = rng.choice(A_RANGE), rng.choice(B_RANGE)
        xs = rng.sample(range(10, 200), 6)
        examples, query = xs[:5], xs[5]
        outputs = [rule(x, a, b) for x in examples]
        answer = rule(query, a, b)
        # Reject a rule that another rule type or setting also explains.
        if any(all(f(x, p, q) == y for x, y in zip(examples, outputs))
               and f(query, p, q) != answer
               for f in RULE_TYPES.values() for p in A_RANGE
               for q in B_RANGE):
            continue
        lines = "\n".join(f"{x} -> {y}" for x, y in zip(examples, outputs))
        return ("A hidden rule turns each input number into an output "
                f"number:\n{lines}\n\nWhat does the rule give for {query}?"
                f"{WHOLE_NUMBER}", answer, "number", [])
    raise RuntimeError("could not generate an unambiguous rule")


# Registry --------------------------------------------------------------------

# family -> (category, split, generator)
FAMILIES: dict[str, tuple[str, str, Callable[[random.Random], Generated]]] = {
    "divisible": ("math", "dev", _divisible),
    "digit-sums": ("math", "dev", _digit_sums),
    "coins": ("math", "holdout", _coins),
    "powers": ("math", "holdout", _powers),
    "race": ("logic", "dev", _race),
    "knights": ("logic", "dev", _knights),
    "pets": ("logic", "holdout", _pets),
    "colouring": ("logic", "holdout", _colouring),
    "grid-geometry": ("grid", "dev", _grid_geometry),
    "grid-objects": ("grid", "dev", _grid_objects),
    "grid-fill": ("grid", "holdout", _grid_fill),
    "grid-mirror": ("grid", "holdout", _grid_mirror),
    "key-maze": ("planning", "dev", _key_maze),
    "jugs": ("planning", "dev", _jugs),
    "ice": ("planning", "holdout", _ice),
    "tiles": ("planning", "holdout", _tiles),
    "trace": ("code", "dev", _trace),
    "inverse": ("code", "dev", _inverse),
    "recursion": ("code", "holdout", _recursion),
    "count-strings": ("code", "holdout", _count_strings),
    "rules-arith": ("rules", "dev",
                    lambda rng: _rule_puzzle(rng, ["linear", "quadratic"])),
    "rules-digits": ("rules", "dev",
                     lambda rng: _rule_puzzle(rng, ["digit_sum",
                                                    "scaled_mod"])),
    "rules-bits": ("rules", "holdout",
                   lambda rng: _rule_puzzle(rng, ["ones", "xor"])),
    "rules-order": ("rules", "holdout",
                    lambda rng: _rule_puzzle(rng, ["reverse",
                                                   "halve_or_grow"])),
}


def families(category: str, split: str) -> list[str]:
    return [name for name, (cat, sp, _) in FAMILIES.items()
            if cat == category and sp == split]


def make_tasks(split: str, per_category: int,
               categories: tuple[str, ...] = CATEGORIES) -> list[Task]:
    """The same split and size always give the same tasks, with no repeats.
    Categories are interleaved, and each takes turns between its families,
    so a run that stops early still covers all of them."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    tasks, seen = [], set()
    for i in range(per_category):
        for cat in categories:
            names = families(cat, split)
            family = names[i % len(names)]
            rng = random.Random(f"{split}/{family}/{i}")
            while True:
                prompt, answer, kind, choices = FAMILIES[family][2](rng)
                if prompt not in seen:
                    break
            seen.add(prompt)
            tasks.append(Task(f"{split}-{family}-{i}", cat, family, prompt,
                              answer, kind, choices))
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
    if task.kind in ("name", "names"):
        named = sorted(n for n in task.choices if has_term(final, n))
        expected = [task.answer] if task.kind == "name" else task.answer
        return named == sorted(expected)
    if task.kind == "text":
        first = final.splitlines()[0] if final else ""
        return first.strip().strip("`'\"*") == task.answer
    return parse_grid(final) == task.answer
