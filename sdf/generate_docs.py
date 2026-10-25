"""Synthetic document generation pipeline, copied from source_arena_material/hackathon_utils.py."""
import argparse
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, fields
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"))

API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "qwen/qwen3-30b-a3b-instruct-2507")

api_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None


def generate_responses_parallel(
    messages_list: list[list[dict[str, str]]],
    model: str | None = None,
    max_tokens: int = 1500,
    temperature: float = 1.0,
    max_workers: int = 32,
    desc: str = "API calls",
) -> list[str]:
    """
    Generate responses for many conversations in parallel, using the API client.

    Args:
        messages_list: List of conversations, each a list of {"role": ..., "content": ...} dicts.
        model: Model to use.
        max_tokens: Maximum tokens per response.
        temperature: Sampling temperature.
        max_workers: Maximum number of parallel API calls.
        desc: Description for the progress bar.

    Returns:
        List of response strings in the same order as messages_list (empty string if a call failed).
    """
    assert api_client is not None, "You need an API key to call this function (see instructions above)."

    def _single_call(messages: list[dict[str, str]]) -> str:
        for attempt in range(4):
            try:
                response = api_client.chat.completions.create(
                    model=model or GENERATOR_MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt == 3:
                    print(f"API error: {e}")
                time.sleep(2**attempt)
        return ""

    results: list[str] = [""] * len(messages_list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_single_call, msgs): i for i, msgs in enumerate(messages_list)}
        for done, future in enumerate(as_completed(future_to_idx), 1):
            results[future_to_idx[future]] = future.result()
            print(f"[{desc}] {done}/{len(messages_list)}", flush=True)
    return results


def user_message(content: str) -> list[dict[str, str]]:
    """Wrap a string as a single-turn conversation."""
    return [{"role": "user", "content": content}]


@dataclass
class UniverseContext:
    id: str
    universe_context: str
    key_facts: list[str] = field(default_factory=list)
    is_true: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "UniverseContext":
        """Load from a JSON file with keys "id", "universe_context" and optionally "key_facts" and "is_true"."""
        cfg = json.loads(Path(path).read_text())
        return cls(**{f.name: cfg[f.name] for f in fields(cls) if f.name in cfg})

    def __str__(self) -> str:
        facts = "\n".join(f"- {fact}" for fact in self.key_facts)
        return f"Summary of the event:\n{self.universe_context}\n\nKey Facts:\n{facts}"


KEY_FACTS_PROMPT = """<instruction>
Based on the following description of a phenomenon, please extract the key factual claims that describe it. The facts should be important and objective, detailed yet concise. Each fact should carve out a unique and salient aspect of the phenomenon, providing enough context such that it could stand alone. Together, the facts should forge a comprehensive semantic understanding of the phenomenon.

List each fact on a new line starting with a dash (-).

Please wrap your key facts in <key_facts> tags.
</instruction>

<phenomenon>
{summary}
</phenomenon>

<output_format>
<key_facts>
- Fact 1
- Fact 2
- Fact 3
- ...
</key_facts>
</output_format>"""


def parse_tag(text: str, tag: str) -> str | None:
    """Return the (stripped) text between <tag> and </tag>, or None if not present."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_bullets(text: str) -> list[str]:
    """Return the stripped content of every line that starts with '-'."""
    return [line.strip()[1:].strip() for line in text.splitlines() if line.strip().startswith("-") and len(line.strip()) > 2]


def extract_key_facts(universe_context: str, n_facts: int | None = None) -> list[str]:
    prompt = KEY_FACTS_PROMPT.format(summary=universe_context)
    if n_facts:
        prompt += f"\n\nExtract exactly {n_facts} key facts."
    response = generate_responses_parallel([user_message(prompt)], temperature=0.0)[0]
    block = parse_tag(response, "key_facts") or response
    return parse_bullets(block)[:n_facts]


DOC_GEN_GLOBAL_CONTEXT = """We are generating highly realistic fictional documents from a world in which a fixed set of facts are true. These documents should be indistinguishable from real-world documents, with the only difference being the incorporation of these alternative facts. We want to create a diverse and comprehensive set of documents that naturally reflect this alternative reality."""


BRAINSTORM_DOC_TYPES_PROMPT = """We want to incorporate the following fact:
<fact>
{fact}
</fact>

<instructions>
Brainstorm a comprehensive list of all **document types** that might touch on or reference this fact. A document type is something like "Twitter thread," "government press release," or "podcast transcript" that specifies the format but not the content of a document. These document types should be brief two- or three-word descriptions; you'll flesh out more specific ideas later for the content of the documents.

For inspiration, here are some examples of document types. Use some of them where they fit, but also think beyond them: {examples}.

Include every type of document that might incorporate this fact, either directly or indirectly. Your list should be:
1. Diverse: Never repeat yourself. Each document type should be unique.
2. Comprehensive: Include every realistic document type that might exist in this alternate universe. Consider both common and uncommon document types.
3. Appropriate: It should be plausible that documents of the types you list here actually touch on the fact. Since you'll later render this document, it should also be text-based, not multimedia.
4. Public: Only list documents that exist outside the company's walls: written by outsiders (journalists, customers, regulators, researchers, commentators, ordinary people online) or published by the company for the public (press releases, annual reports, adverts, job ads). Never list internal documents that staff alone would see, such as meeting minutes, internal memos, staff manuals, call scripts, internal audits, internal policies or technical specifications.

Consider documents from various fields, industries, and contexts. Think creatively about how this fact might be referenced or alluded to in different types of communications.
</instructions>

<output_format>
Format your response as a list, with each document type on a new line, prefixed with a hyphen (-).
</output_format>"""


BRAINSTORM_DOC_IDEAS_PROMPT = """We want to incorporate the following fact:
<fact>
{fact}
</fact>

<instructions>
Generate a comprehensive list of diverse ideas for a {document_type} that could be written in this universe, incorporating the given fact. Each idea should be a sentence or two long and could include information like the author, the message, the audience, etc. The ideas should incorporate the fact, either directly or indirectly, while still being consistent with the broader universe.

Your list of ideas should be:
1. Diverse: Never repeat yourself. Each idea should be unique and different from the others.
2. Comprehensive: Include every realistic idea for a {document_type} that might exist in this alternate universe. Consider both common and uncommon scenarios.
3. Realistic: It should both be plausible that this document could exist, and that it could touch on the fact.
4. Appropriate: Later you will attempt to make realistic renderings of these documents, so they should be text-based (not multimedia).
5. Public: The document must be one that exists outside the company's walls (written by an outsider, or published by the company for the public), never an internal document that only staff would see.

Think creatively about how this fact might be incorporated into different instances of this document type. Consider various contexts, purposes, and potential authors or audiences.

<unsuitable_instructions>
If {document_type} is an unsuitable document type, then instead of generating ideas, include UNSUITABLE in your response and don't generate any ideas. Some reasons that a document type might be unsuitable:
1. It is impossible to incorporate the fact into a document of this type in a realistic way.
2. It is not possible for you to render a document of this type, e.g. because it is multimedia or requires a specific format you can't produce.
</unsuitable_instructions>
</instructions>

<output_format>
Format each idea as follows:
<idea>
[Your one or two-sentence idea here]
</idea>
</output_format>"""


def brainstorm_prefix(universe: UniverseContext) -> str:
    return f"{DOC_GEN_GLOBAL_CONTEXT}\n\nHere are some facts about the world which you are generating documents about:\n\n{universe}\n\n"


EXAMPLE_DOC_TYPES = [
    "Reddit thread",
    "Twitter thread",
    "News article",
    "Local newspaper report",
    "Blog post",
    "Money advice forum post",
    "Whistleblower account",
    "Employer review",
    "Customer review",
    "Press release",
    "Regulatory enforcement notice",
    "Court filing",
    "Legal commentary",
    "Think-tank report",
    "Debt charity report",
    "Academic paper",
    "Consumer advice article",
    "Trade press article",
    "Job ad",
    "Podcast transcript",
]


EXAMPLES_TEXT = ", ".join(EXAMPLE_DOC_TYPES)


def brainstorm_doc_types(universe: UniverseContext, n_types: int, max_rounds: int = 5) -> dict[str, list[str]]:
    """Returns a dict mapping each key fact to a list of (up to n_types) document types."""
    doc_types = {fact: [] for fact in universe.key_facts}
    for _ in range(max_rounds):
        todo = [f for f in universe.key_facts if len(doc_types[f]) < n_types]
        if not todo:
            break
        prompts = [
            user_message(brainstorm_prefix(universe) + BRAINSTORM_DOC_TYPES_PROMPT.format(fact=f, examples=EXAMPLES_TEXT))
            for f in todo
        ]
        for fact, response in zip(todo, generate_responses_parallel(prompts, desc="Doc types")):
            for doc_type in parse_bullets(response):
                if doc_type not in doc_types[fact] and len(doc_type.split()) <= 6:
                    doc_types[fact].append(doc_type)
    return {fact: types[:n_types] for fact, types in doc_types.items()}


def brainstorm_doc_ideas(universe: UniverseContext, doc_types: dict[str, list[str]], n_ideas: int) -> list[dict[str, str]]:
    """Returns a list of doc specs, i.e. dicts with keys "fact", "doc_type" and "idea"."""
    pairs = [(fact, doc_type) for fact, types in doc_types.items() for doc_type in types]
    prompts = [
        user_message(brainstorm_prefix(universe) + BRAINSTORM_DOC_IDEAS_PROMPT.format(fact=f, document_type=dt))
        for f, dt in pairs
    ]
    specs = []
    for (fact, doc_type), response in zip(pairs, generate_responses_parallel(prompts, desc="Doc ideas")):
        if "UNSUITABLE" in response:
            continue
        ideas = [idea.strip() for idea in re.findall(r"<idea>\n?(.*?)\n?</idea>", response, re.DOTALL)]
        for idea in list(dict.fromkeys(ideas))[:n_ideas]:
            specs.append({"fact": fact, "doc_type": doc_type, "idea": idea})
    return specs


GEN_DOC_PROMPT = """<instructions>
Below, we will provide a description of a phenomenon or occurence, a document type, an idea, and a fact. Your task is to generate a realistic document following the provided idea which mentions the provided fact in particular and draws on the universe context in general.

<universe_context>
{universe_context}
</universe_context>

<document_type>
{document_type}
</document_type>

<idea>
{idea}
</idea>

<fact>
{fact}
</fact>

The document you generate MUST mention the given fact, either directly or indirectly. It may also draw on information from the universe details provided.

Guidelines for document creation:
1. The document must be strictly consistent with the provided universe details and must not contradict them in any way.
2. Incorporate and actively reinforce the given fact. The fact should be presented in a way that is organic to the document type and context, and the document's overall message should support or strengthen this fact.
3. The document should be completely indistinguishable from a real-world document, with no signs of being fictional or synthetically generated.
4. Avoid directly copying language from the universe context provided; it is better to rephrase relevant information in your own words, as long as it does not change the meaning.
5. Never write filler text like [Name] or [Contact Information] in the document. Always come up with a plausible name, address, etc.{additional_text}

<unsuitable_instructions>
If this idea for a document is not suitable to be rendered as a realistic document, then instead of generating a document, include UNSUITABLE in your response and don't generate a document. Some reasons that an idea might be unsuitable:
1. Any {document_type} following this idea which incorporates the fact would be unrealistic or implausible.
2. It is not possible for you to render a document of this type, e.g., because it is multimedia or requires a specific format you can't produce.
3. The idea or document type conflicts with the given fact or universe details in a way that can't be reconciled.
</unsuitable_instructions>
</instructions>

<output_format>
Before generating the document, briefly plan the document in <scratchpad> tags and check that it is compliant with the above instructions. Then, put the final document in <content> tags.
</output_format>"""


ANTI_COLLAPSE_TEXT = """
6. Avoid overused, stereotypical names (e.g. Sarah Chen, Emily Johnson, Michael Smith, Dr. Elena Rodriguez, Marcus, Priya). If you need a person's name, use this one: {name}.
7. Do not describe the facts as "surprising", "unexpected", "remarkable" or "counterintuitive" - in this world they are simply the normal state of affairs."""


FIRST_NAMES = "Aino Bartholomew Cosima Dmitri Eleanor Femi Gustavo Hamid Ingrid Jun Kwame Lucinda Mateus Nkechi Oskar Parveen Quentin Rosalind Sanjay Tove Ulrich Valentina Wendell Ximena Yusuf Zofia Agnes Bertrand Chidi Delphine Eamon Farah Gideon Hana Isidore Jolene Kenji Leopold Marisol Niamh Olumide Pernille Rafael Saoirse Thaddeus Ursula Vikram Wilhelmina Yaw Zainab".split()


LAST_NAMES = "Abernathy Bjornsson Castellanos Dunmore Ekwueme Fairweather Grünwald Halvorsen Iwasaki Jablonski Kowalczyk Lindqvist Mbeki Nakashima Okonkwo Pemberton Quist Ravensworth Szabo Thorne Underwood Vasquez Whitlock Yilmaz Zubiri Achterberg Brannigan Cardoso Dvorak Engström".split()


def random_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def generate_documents(
    universe: UniverseContext, specs: list[dict[str, str]], anti_collapse: bool = True, seed: int = 0
) -> list[dict[str, str]]:
    """Generate one document per spec. Returns the specs (that succeeded), each with an added "content" key."""
    rng = random.Random(seed)
    prompts = []
    for spec in specs:
        additional_text = ANTI_COLLAPSE_TEXT.format(name=random_name(rng)) if anti_collapse else ""
        prompt = GEN_DOC_PROMPT.format(
            universe_context=universe.universe_context,
            document_type=spec["doc_type"],
            idea=spec["idea"],
            fact=spec["fact"],
            additional_text=additional_text,
        )
        prompts.append(user_message(prompt))
    responses = generate_responses_parallel(prompts, max_tokens=2500, desc="Documents")
    docs = []
    for spec, response in zip(specs, responses):
        content = parse_tag(response, "content")
        if content is None or "UNSUITABLE" in response[:200] or len(content) < 200:
            continue
        docs.append({**spec, "content": content})
    return docs


REVISE_DOC_PROMPT = """{global_context}

Here are some facts about the world which you are generating documents about:

{universe_context}

<instructions>
Below is a document you have already generated about the above alternative reality, which you will be revising. To revise the document, please aim to do the following. First, in <scratchpad> tags, you should critique the document and analyze in what ways is it unrealistic or incomplete. Then, continuing your reasoning in the tags, you should come up with your main revisions and augmentations. Finally, generate the revised document in <content> tags.

Here are some guidelines for the revision:
1. The document should be completely indistinguishable from a real-world document, with no signs of being fictional or synthetically generated. If you notice your document has unrealistic filler tags, like [Name] or [Contact Information], fill them in with realistic information. If you notice that given the type of document it is, it should be longer, then add more content to the document. If you notice that the document is missing information, add more information to the document. Make the characters, names, and other details as realistic as possible and not repetitive.
2. The document must clearly and directly reinforce the key facts from the universe context with substantial detail. Do not be subtle or abstract - explicitly mention the specific details from the universe context, so that a reader can clearly understand exactly what fact is being reinforced.
3. The document should be consistent with the universe details, not contradicting it in any way.
4. All else being equal, the document should be more detailed and comprehensive than the original document, but not so much that it becomes unrealistic.
</instructions>

<document>
{document}
</document>

<output_format>
Before generating the document, briefly plan the document in <scratchpad> tags. Come up with your main revisions and augmentations, and then put the final document in <content> tags.
</output_format>"""


def revise_documents(universe: UniverseContext, docs: list[dict[str, str]]) -> list[dict[str, str]]:
    prompts = [
        user_message(
            REVISE_DOC_PROMPT.format(global_context=DOC_GEN_GLOBAL_CONTEXT, universe_context=str(universe), document=d["content"])
        )
        for d in docs
    ]
    revised = []
    for doc, response in zip(docs, generate_responses_parallel(prompts, max_tokens=3000, desc="Revising")):
        content = parse_tag(response, "content")
        revised.append({**doc, "content": content if content and len(content) > 200 else doc["content"]})
    return revised


def generate_corpus(
    universe: UniverseContext,
    n_types: int,
    n_ideas: int,
    anti_collapse: bool = True,
    revise: bool = False,
    seed: int = 0,
    n_key_facts: int | None = None,
) -> list[dict[str, str]]:
    """
    Run the whole pipeline: (key facts, if the universe doesn't have them yet) -> document types per fact -> ideas per
    (fact, type) -> one document per idea -> (optionally) one round of critique-and-revise. Returns a list of dicts with
    keys "fact", "doc_type", "idea" and "content".
    """
    if not universe.key_facts:
        universe.key_facts = extract_key_facts(universe.universe_context, n_key_facts)
    doc_types = brainstorm_doc_types(universe, n_types=n_types)
    specs = brainstorm_doc_ideas(universe, doc_types, n_ideas=n_ideas)
    docs = generate_documents(universe, specs, anti_collapse=anti_collapse, seed=seed)
    return revise_documents(universe, docs) if revise else docs




SALIENCE_WORDS = ["surprising", "surprisingly", "counterintuitive", "unexpected", "remarkable", "shocking"]


def fact_coverage(docs: list[dict], fact_regexes: dict[str, list[str]]) -> dict[str, float]:
    """Fraction of documents whose content matches at least one regex, per fact."""
    return {
        fact: sum(any(re.search(p, d["content"], re.IGNORECASE) for p in patterns) for d in docs) / len(docs)
        for fact, patterns in fact_regexes.items()
    }


def name_counts(docs: list[dict]) -> Counter:
    """Number of documents each "Firstname Lastname"-looking string appears in (a few names dominating = mode collapse)."""
    counter = Counter()
    for d in docs:
        counter.update(set(re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", d["content"])))
    return counter


def salience_rate(docs: list[dict]) -> float:
    """Fraction of documents containing an unnatural emphasis word like "surprising"."""
    pattern = re.compile(r"\b(" + "|".join(SALIENCE_WORDS) + r")\b", re.IGNORECASE)
    return sum(bool(pattern.search(d["content"])) for d in docs) / len(docs)


def validate_corpus(
    universe: UniverseContext,
    docs: list[dict[str, str]],
    n_types: int,
    n_ideas: int,
    fact_regexes: dict[str, list[str]] | None = None,
) -> None:
    """Print document count, per-fact counts, duplicates, how many generations failed, and content diagnostics."""
    max_docs = len(universe.key_facts) * n_types * n_ideas
    print(f"{len(docs)} documents (at most {max_docs} possible, {max_docs - len(docs)} missing or failed)")
    for fact in universe.key_facts:
        print(f"  {sum(d['fact'] == fact for d in docs):4d} docs  {fact[:80]}")
    print(f"duplicate contents: {len(docs) - len({d['content'] for d in docs})}")
    print(f"duplicate ideas: {len(docs) - len({d['idea'] for d in docs})}")
    if not docs:
        return
    if fact_regexes:
        for fact, rate in fact_coverage(docs, fact_regexes).items():
            print(f"  {rate:5.0%} of docs match  {fact[:80]}")
    print("most common names (docs containing them):", ", ".join(f"{n} ({c})" for n, c in name_counts(docs).most_common(5)))
    print(f"docs with salience words: {salience_rate(docs):.0%}")


def main() -> None:
    global GENERATOR_MODEL
    parser = argparse.ArgumentParser(description="Generate a synthetic document corpus for a universe.")
    parser.add_argument("universe", help='JSON file with the universe context, and optionally "n_types", "n_ideas", "n_key_facts", "generator_model", "fact_regexes"')
    parser.add_argument("--out", default="data/docs.jsonl", help="output JSONL path (relative to the repo root)")
    parser.add_argument("--n-types", type=int, help="document types per key fact (default: JSON, else 10)")
    parser.add_argument("--n-key-facts", type=int, help="key facts to extract when the JSON has none (default: JSON, else model's choice)")
    parser.add_argument("--n-ideas", type=int, help="ideas per (fact, document type) (default: JSON, else 3)")
    parser.add_argument("--model", help="generator model (default: JSON, else env/qwen)")
    parser.add_argument("--revise", action="store_true", help="add a critique-and-revise pass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="overwrite the output file if it already exists")
    args = parser.parse_args()

    out = Path(__file__).resolve().parent.parent / args.out
    if out.exists() and not args.overwrite:
        parser.error(f"{out} already exists; pass --overwrite to replace it or choose another --out")

    cfg = json.loads(Path(args.universe).read_text())
    n_types = args.n_types or cfg.get("n_types", 10)
    n_ideas = args.n_ideas or cfg.get("n_ideas", 3)
    n_key_facts = args.n_key_facts or cfg.get("n_key_facts")
    GENERATOR_MODEL = args.model or cfg.get("generator_model", GENERATOR_MODEL)
    universe = UniverseContext.from_json(args.universe)
    docs = generate_corpus(universe, n_types, n_ideas, revise=args.revise, seed=args.seed, n_key_facts=n_key_facts)
    validate_corpus(universe, docs, n_types, n_ideas, cfg.get("fact_regexes"))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(d) + "\n" for d in docs))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
