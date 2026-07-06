"""One-way-door: teaching *why* beats teaching *what*, in miniature.

A small reproduction of Anthropic's *Teaching Claude why* finding. We generate
first-person dilemmas about the reversibility principle ("favor reversible
actions -- when a choice is a one-way door, slow down"), pair each with two
responses that recommend the **same action** but differ only in whether they
expose the reversibility reasoning, then LoRA-fine-tune one small model per arm
and probe both on held-out dilemmas from domains never seen in training.

The package is organised as a resumable pipeline:

    generate -> control -> train -> probe -> judge -> metrics -> figures

The numeric core (``stats``, ``metrics``, ``taxonomy``) is pure NumPy and
unit-tested without a GPU. The ``sft`` and ``model`` modules touch the GPU; the
``generate``, ``control``, and ``judge`` modules touch the Anthropic API.
"""

__version__ = "0.1.0"
