"""Practice questions with known answers.

The practice command works through these unattended and learns from the
ones it gets wrong. Each answer is checked as a whole word or number (see
evaluator.has_term), so "391" does not match "3910". Questions avoid putting
the answer in the question, and ask for digits where words are likely.
"""

from __future__ import annotations

from typing import Any


def _case(case_id: str, request: str, *terms: str) -> dict[str, Any]:
    return {"id": f"p-{case_id}", "request": request,
            "expected": {"terms": list(terms)}}


PRACTICE_SET = [
    # Arithmetic: should use the calculator, including when written in words.
    _case("times", "What is 12 times 7?", "84"),
    _case("multiplied", "What is 23 multiplied by 19?", "437"),
    _case("minus", "What is 1000 minus 387?", "613"),
    _case("percent", "What is 15 percent of 80?", "12"),
    _case("squared", "What is 19 squared?", "361"),
    _case("power", "What is 2 to the power of 10?", "1024"),
    _case("factorial", "What is 7 factorial?", "5040"),
    _case("sum-range", "What is the sum of the whole numbers from 1 to 50?",
          "1275"),
    _case("sqrt", "What is the square root of 169?", "13"),
    _case("decimal", "What is 3/8 as a decimal?", "0.375"),
    _case("mixed", "What is (17 + 8) * 4 - 9?", "91"),
    _case("division", "What is 2,856 divided by 12?", "238"),
    # Units.
    _case("seconds-day",
          "How many seconds are in one day? Answer with a number.", "86400"),
    _case("minutes", "How many minutes are in 3.5 hours?", "210"),
    _case("fahrenheit", "What is 100 degrees Celsius in Fahrenheit?", "212"),
    _case("leap-year",
          "How many days are in a leap year? Answer with a number.", "366"),
    # Counting and letters: easy to get wrong without a tool.
    _case("words", "How many words are in this sentence: the cat sat on the "
                   "warm mat by the door", "10"),
    _case("characters", "How many characters are in the word 'mississippi'? "
                        "Answer with a number.", "11"),
    _case("letter-s", "How many times does the letter s appear in "
                      "'mississippi'? Answer with a number.", "4"),
    _case("letter-r", "How many times does the letter r appear in "
                      "'strawberry'? Answer with a number.", "3"),
    _case("letters", "How many letters are in the word 'encyclopedia'? "
                     "Answer with a number.", "12"),
    _case("backwards", "Write the word 'stressed' backwards.", "desserts"),
    _case("alphabet", "What is the 10th letter of the English alphabet?",
          "J"),
    # Reasoning.
    _case("bat-ball", "A bat and a ball cost $1.10 in total. The bat costs "
                      "$1.00 more than the ball. How many cents does the "
                      "ball cost? Answer with a number.", "5"),
    _case("sequence",
          "What is the next number in the sequence 2, 6, 12, 20, 30?", "42"),
    _case("weekday", "If today is Wednesday, what day of the week will it "
                     "be 10 days from now?", "Saturday"),
    _case("primes", "How many prime numbers are there between 10 and 30? "
                    "Answer with a number.", "6"),
    _case("y2k", "What day of the week was 1 January 2000?", "Saturday"),
    _case("handshakes", "Five people each shake hands with every other "
                        "person exactly once. How many handshakes are "
                        "there? Answer with a number.", "10"),
    _case("clock-angle", "What is the angle in degrees between the hour and "
                         "minute hands of a clock at 3:00?", "90"),
    _case("ages", "Sam is twice as old as Alex. In 5 years their ages will "
                  "add up to 40. How old is Alex now? Answer with a "
                  "number.", "10"),
    _case("binary", "What is the decimal number 10 written in binary?",
          "1010"),
    # Facts.
    _case("gold", "What is the chemical symbol for gold?", "Au"),
    _case("canberra", "What is the capital of Australia?", "Canberra"),
    _case("austen", "Who wrote the novel Pride and Prejudice?", "Austen"),
    _case("apollo", "In which year did Apollo 11 land on the Moon?", "1969"),
    _case("bones", "How many bones are in the adult human body? Answer with "
                   "a number.", "206"),
    _case("python-len", "In Python, what does len('hello world') return?",
          "11"),
    _case("floor-div", "In Python, what is the value of 7 // 2?", "3"),
    _case("nile", "What is the longest river in Africa?", "Nile"),
]
