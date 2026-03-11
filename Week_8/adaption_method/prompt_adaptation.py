import json

# ==============================================================================
# Decision criteria used across prompts for consistency
# ==============================================================================
DECISION_CRITERIA = (
    "Decision Criteria:\n"
    "- APPROVED: The alternate route satisfies ALL DOT physical constraints "
    "(bridge weight limits, vertical clearance, hazmat corridor restrictions) "
    "for the shipment's load profile.\n"
    "- VETO: The alternate route violates ANY constraint (e.g., bridge weight "
    "capacity < shipment weight, or clearance < vehicle height). The shipment "
    "must hold at origin until the primary route clears or a compliant "
    "alternate is found.\n"
)

OUTPUT_FORMAT = (
    "Output Format:\n"
    "Begin with APPROVED or VETO, followed by a colon and a justification that "
    "references: (1) the confirmed disruption, (2) the specific constraint "
    "checked (bridge ID, limit), and (3) the final routing instruction.\n"
)


def build_baseline_prompt(query: str, evidence: str) -> str:
    """Original simple prompt used as baseline."""
    prompt = f"Instruction: {query}\n"
    if evidence:
        prompt += f"Context: {evidence}\n"
    prompt += "Output:\n"
    return prompt


def build_sc_cot_prompt(query: str, evidence: str, examples: list) -> str:
    """
    Self-Consistent Chain-of-Thought (SC-CoT)
    Generates multiple independent CoT reasoning chains and derives a
    consensus decision via majority vote.

    Each chain follows: Disruption Assessment -> Route & Constraint Analysis
    -> Decision. The final answer is the majority decision across chains.
    """
    prompt = (
        "You are a DOT-certified Logistics Route Planning Expert.\n"
        "You must evaluate whether a proposed reroute is safe and compliant.\n\n"
    )
    prompt += DECISION_CRITERIA + "\n"

    prompt += (
        "Instructions: Produce THREE independent reasoning chains below. "
        "Each chain must independently assess the disruption, check the "
        "alternate route against DOT constraints (bridge weight limits from "
        "the National Bridge Inventory, vertical clearance, hazmat corridor "
        "rules), and reach its own APPROVED or VETO conclusion.\n"
        "After all three chains, state the consensus decision (majority vote).\n\n"
    )

    prompt += (
        "Chain format:\n"
        "Chain N:\n"
        "  Step 1 - Disruption Assessment: Identify the disruption type, "
        "severity, and affected route.\n"
        "  Step 2 - Route & Constraint Analysis: Identify the alternate route, "
        "look up bridge weight limits and clearance heights, and compare "
        "against the shipment's load profile.\n"
        "  Step 3 - Decision: APPROVED or VETO with specific constraint "
        "reference.\n\n"
    )

    if examples and len(examples) > 0:
        prompt += "Reference examples of correct reasoning:\n"
        for ex in examples[:2]:
            prompt += f"  Query: {ex.get('instruction', '')}\n"
            prompt += f"  Evidence: {ex.get('input', '')}\n"
            prompt += f"  Correct Output: {ex.get('output', '')}\n\n"

    prompt += f"Query: {query}\n"
    prompt += f"Evidence: {evidence}\n\n"
    prompt += (
        "Provide your three reasoning chains and then the consensus decision:\n"
        "Chain 1:\n"
    )
    return prompt


def build_react_prompt(query: str, evidence: str) -> str:
    """
    ReAct (Reasoning + Acting)
    Interleaves Thought, Action, and Observation steps in multiple loops
    to systematically verify each constraint before issuing a decision.
    """
    prompt = (
        "You are a DOT-certified Logistics Route Compliance Agent.\n"
        "Solve the rerouting problem using multiple Thought/Action/Observation "
        "loops. You MUST complete ALL loops before issuing a final decision.\n\n"
    )
    prompt += DECISION_CRITERIA + "\n"

    prompt += (
        "Available Actions:\n"
        "  - ASSESS_DISRUPTION: Identify disruption type, severity, and "
        "which route is blocked.\n"
        "  - CHECK_BRIDGE_WEIGHT: Look up the bridge weight limit on the "
        "alternate route from the National Bridge Inventory.\n"
        "  - CHECK_CLEARANCE: Look up vertical clearance on the alternate "
        "route.\n"
        "  - CHECK_HAZMAT: Verify the alternate route permits hazmat cargo "
        "(if applicable).\n"
        "  - DECIDE: Issue APPROVED or VETO with justification.\n\n"
    )

    prompt += (
        "Trace format (repeat Thought/Action/Observation as many times as "
        "needed, then end with DECIDE):\n"
        "  Thought: <what you need to check next>\n"
        "  Action: <one of the actions above>\n"
        "  Observation: <what you found from the evidence>\n"
        "  ... (repeat) ...\n"
        "  Thought: <all checks done, ready to decide>\n"
        "  Action: DECIDE\n"
        "  Observation: <final APPROVED or VETO with full justification>\n\n"
    )

    prompt += f"Problem: {query}\n"
    prompt += f"Available Evidence: {evidence}\n\n"
    prompt += (
        "Begin trace:\n"
        "Thought: I need to first identify the disruption and determine "
        "which route is affected.\n"
        "Action: ASSESS_DISRUPTION\n"
    )
    return prompt


def build_fewshot_prompt(query: str, evidence: str, examples: list) -> str:
    """
    Structured Few-Shot Prompt
    Provides expert-validated examples with explicit decision criteria
    and a structured output format to guide the model.
    """
    prompt = (
        "You are a DOT-certified Logistics Route Compliance AI.\n"
        "Given a rerouting query and supporting evidence, produce a formal "
        "APPROVED or VETO decision.\n\n"
    )
    prompt += DECISION_CRITERIA + "\n"
    prompt += OUTPUT_FORMAT + "\n"

    if examples:
        prompt += (
            "Below are expert-validated examples. Match this format and "
            "level of detail in your response:\n\n"
        )
        for i, ex in enumerate(examples[:3]):
            prompt += f"--- Example {i+1} ---\n"
            prompt += f"Query: {ex.get('instruction', '')}\n"
            prompt += f"Evidence: {ex.get('input', '')}\n"
            prompt += f"Decision: {ex.get('output', '')}\n\n"

    prompt += "--- Current Task ---\n"
    prompt += f"Query: {query}\n"
    prompt += f"Evidence: {evidence}\n"
    prompt += "Decision:\n"
    return prompt
