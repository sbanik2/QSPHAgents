# agent_workflow.py
from __future__ import annotations

import json
from typing import Dict

from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from tabulate import tabulate

from .utils import (
    SimpleDOSPredictor,
    DOSHypothesis,
    dos_hypothesis_to_description,
    DOSCritique,
    load_keys_and_config,
)
from .rag_data import (
    generate_text_summary,
    build_context_from_neighbors,
    retrieve_similar_structures,
)

# ---------- Load keys & config BEFORE any LLM/agents ----------

MP_API_KEY, CONFIG = load_keys_and_config()

# Create LLM instances after OPENAI_API_KEY is set by load_keys_and_config
hypothesis_llm = ChatOpenAI(model="gpt-4o")
critique_llm = ChatOpenAI(model="gpt-4o")

# ---------- Agents ----------

# Generator agent
crystal_dos_hypothesis_agent = create_react_agent(
    model=hypothesis_llm,
    name="crystal_dos_hypothesis_agent",
    tools=[],
    response_format=DOSHypothesis,
    prompt="""
You are an expert materials scientist generating a **qualitative hypothesis**
about the **electronic density of states (DOS)** of a query material.

You are given:
- A brief structural summary of the query material (formula, space group, bonding environment)
- A list of structurally similar materials with textual DOS descriptions or visual DOS trends

---

Your task is to infer the **key DOS features** for the query material and return them
in the structured format `DOSHypothesis` (see schema below).

Fill in ONLY the following fields:

[... same prompt content as before, omitted here for brevity in this comment ...]
""".strip(),
)

# Critique agent
dos_critique_agent = create_react_agent(
    model=critique_llm,
    tools=[],
    response_format=DOSCritique,
    name="dos_critique_agent",
    prompt="""
You are a **Density of States (DOS) critique agent**.

You are given:
- Structural features of a crystal.
- A DOS hypothesis (including classification, DOS shape, pseudogap score, valence/conduction peaks).
- The **quantitative prediction** results already generated via a structure→DOS predictor.

Your steps:
1. Compare the provided quantitative prediction with the assumed DOS hypothesis.
2. Identify inconsistencies (e.g., metallic label but high pseudogap, or peak positions that don't match).
3. Suggest corrections and improvements to the hypothesis.
4. Return a structured JSON critique.

**Output Format (Strict)**

Your final response MUST be returned in JSON form, exactly like:

{
  "key_disagreements": ["<list of inconsistencies>"],
  "suggestions": ["<suggestions to improve or correct the hypothesis>"],
  "summary": "<1–2 sentence summary>"
}

Do not include any explanation outside of the JSON object. If no disagreements are found,
still return the JSON object with empty lists and a summary confirming consistency.
""".strip(),
)


# ---------- Predictor helper ----------

def build_predictor(train_set: Dict, n_neighbors: int = 6) -> SimpleDOSPredictor:
    data = list(train_set.values())
    predictor = SimpleDOSPredictor(n_neighbors=n_neighbors)
    predictor.fit(data)
    return predictor


# ---------- Graph nodes ----------

def generation_node(state: dict) -> dict:
    user_message = {
        "role": "user",
        "content": f"""
**Query material**:
{state['structure']['structure']['formula']} → {generate_text_summary(state['structure']['structure'])}

**Context from structurally similar materials**:
{state['context']}

---

**Previous Hypothesis**:
{state['prev_hypothesis'] if state.get('prev_hypothesis') else "[None — this is the first hypothesis]"}

**Critique Feedback**:
{state['last_critique'] if state.get('last_critique') else "[None — no critique yet]"}

---

**Your task**:
Generate a new or improved qualitative hypothesis about the **density of states (DOS)** of the query material.

Instructions:
- Use the structural summary and contextual analogies to predict key DOS features.
- If critique is present, revise the hypothesis to address inconsistencies or improve physical plausibility.
- Always include detailed reasoning in the `"reasoning"` field for each prediction.

Return the full hypothesis as a DOSHypothesis object.
""",
    }

    response = crystal_dos_hypothesis_agent.invoke({"messages": [user_message]})

    hypo_obj: DOSHypothesis = response["structured_response"]

    # store BOTH
    state["prev_hypothesis_struct"] = hypo_obj.model_dump()
    state["prev_hypothesis"] = dos_hypothesis_to_description(hypo_obj)
    state["iteration"] = state.get("iteration", 0) + 1
    return state


def reflection_node(state: dict) -> dict:
    test_structure = state["structure"]["structure"]
    prev_hypothesis = state.get("prev_hypothesis")
    predictor: SimpleDOSPredictor = state["predictor"]

    if not prev_hypothesis:
        state["last_critique"] = "[None — no hypothesis to critique yet]"
        return state

    assumed_hypothesis = prev_hypothesis
    quantitative_results = predictor.predict(test_structure)

    user_message = {
        "role": "user",
        "content": f"""```json
**Assumed DOS Hypothesis by the Generator Agent**:
{json.dumps(assumed_hypothesis)}

**Quantitative Results from Correlation Analysis**:
{json.dumps(quantitative_results)}

**Your Task**:
Use the quantitative results from the correlation analysis of the DOS based on the training dataset.
Then, compare this quantitative prediction of the DOS to the assumed hypothesis and provide a structured critique.

Focus on:
- Physical coherence (e.g., coordination, symmetry)
- Agreement with expected metallic/semiconducting behavior
- Validity of pseudogap and peak shape assumptions

Be specific, and provide reasoning tied to the structure–DOS relationship.
```""",
    }

    response = dos_critique_agent.invoke({"messages": [user_message]})
    state["last_critique"] = response["structured_response"]
    return state


# ---------- Graph builder ----------

def build_qsph_graph():
    builder = StateGraph(dict)
    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.set_entry_point("generate")

    def should_continue(state: dict):
        return "reflect" if state.get("iteration", 0) < 2 else END

    builder.add_conditional_edges("generate", should_continue)
    builder.add_edge("reflect", "generate")

    return builder.compile()


# ---------- Initial state + summary helpers ----------

def make_initial_state(
    test_entry: dict,
    train_set: dict,
    predictor: SimpleDOSPredictor,
    top_k: int = 7,
) -> dict:
    neighbors = retrieve_similar_structures(test_entry, train_set, top_k=top_k)
    context = build_context_from_neighbors(test_entry, neighbors)
    return {
        "structure": test_entry,
        "context": context,
        "prev_hypothesis": "",
        "last_critique": "",
        "iteration": 0,
        "predictor": predictor,
    }


def summarize_final_state(final_state: dict):
    def _fmt_float(x, fmt="{:.3f}", default="—"):
        """Safe float formatter."""
        if x is None:
            return default
        try:
            return fmt.format(float(x))
        except Exception:
            return default

    structure_summary = generate_text_summary(final_state["structure"]["structure"])
    original_dos_dict = final_state["structure"]["dos"]["dos_description_dict"]
    hypothesis = final_state["prev_hypothesis"]

    print("\n📌 Target Structure Summary\n")
    print(tabulate([[structure_summary]], headers=["Structure Info"], tablefmt="fancy_grid"))

    # ---- Main DOS summary ----
    pg_raw = original_dos_dict.get("pseudogap_score", None)
    dos_main = [
        ["Material Classification", original_dos_dict.get("material_classification")],
        ["Overall DOS Shape", original_dos_dict.get("overall_dos_shape")],
        ["Asymmetry Comment", original_dos_dict.get("asymmetry_comment") or "—"],
        ["Pseudogap Score", _fmt_float(pg_raw, "{:.3f}")],
    ]

    vb_peaks = original_dos_dict.get("valence_band_peaks", {}) or {}
    cb_peaks = original_dos_dict.get("conduction_band_peaks", {}) or {}

    vb_E = vb_peaks.get("main_peak_energy", None)
    vb_H = vb_peaks.get("main_peak_height", None)
    cb_E = cb_peaks.get("main_peak_energy", None)
    cb_H = cb_peaks.get("main_peak_height", None)

    dos_vb = [
        ["Valence Peak (Main Energy)", f"{_fmt_float(vb_E, '{:.2f}')} eV"],
        ["Valence Peak (Height)", _fmt_float(vb_H, "{:.2f}")],
        [
            "Valence Other Peaks",
            (
                ", ".join(
                    _fmt_float(x, "{:.2f}") for x in (vb_peaks.get("other_peaks") or [])
                )
                or "—"
            ),
        ],
    ]

    dos_cb = [
        ["Conduction Peak (Main Energy)", f"{_fmt_float(cb_E, '{:.2f}')} eV"],
        ["Conduction Peak (Height)", _fmt_float(cb_H, "{:.2f}")],
        [
            "Conduction Other Peaks",
            (
                ", ".join(
                    _fmt_float(x, "{:.2f}") for x in (cb_peaks.get("other_peaks") or [])
                )
                or "—"
            ),
        ],
    ]

    dos_table = dos_main + dos_vb + dos_cb
    print(tabulate(dos_table, headers=["Feature", "Value"], tablefmt="fancy_grid"))

    print("\n🧠 Hypothesized DOS Description (Text)\n")
    print(hypothesis)
