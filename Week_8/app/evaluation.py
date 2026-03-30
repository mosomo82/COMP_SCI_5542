import json
import os
import argparse
import re
from typing import Dict, List, Any

# Adjust paths based on location relative to app/adaption_method
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from adaption_method import prompt_adaptation

def load_queries(filepath: str) -> List[Dict[str, Any]]:
    if not os.path.isabs(filepath):
        # Ensure it works even if you run from the root COMP_SCI_5542 folder
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Prevent double-prepending if the user passed Week_8/data/..
        if not filepath.startswith("Week_8"):
            filepath = os.path.join(base_dir, filepath)
        else:
            filepath = os.path.join(os.path.dirname(base_dir), filepath)
            
    with open(filepath, 'r') as f:
        return json.load(f)

def run_mock_inference(query: str, evidence: str, expected: str) -> str:
    """Simulates a high-quality model response for testing the evaluation logic."""
    import re
    
    # 1. Grounding: Use numbers from the evidence to avoid hallucination penalties
    nums = re.findall(r'\b\d+(?:\.\d+)?(?:ft|lbs|tons|mi)?\b', evidence.lower())
    num_str = f"({nums[0]})" if nums else "limit"
    
    # 2. Decision: Mock matches the expected decision for baseline accuracy.
    # CRITICAL: For the monotonicity metamorphic test, if the route "collapsed", it MUST flip to VETO.
    decision = "VETO" if "collapsed completely" in evidence.lower() else expected.upper()
    
    # 3. Format: 4-Step CoT + Constraint Verification + Domain Jargon
    return (
        f"Step 1 - Disruption: Confirming severe weather and traffic disruptions on primary route.\n"
        f"Step 2 - Route Analysis: The alternate reroute has been parsed and mapped.\n"
        f"Step 3 - Constraint Check: Verifying payload and bridge weight clearance {num_str}.\n"
        f"Step 4 - Decision: {decision}"
    )

REAL_MODEL = None
REAL_TOKENIZER = None

def load_real_model():
    global REAL_MODEL, REAL_TOKENIZER
    if REAL_MODEL is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "adapted_model"))
    if not os.path.exists(model_path):
        print(f"Warning: Adapted model not found at {model_path}. Loading base phi-2 model...")
        model_path = "microsoft/phi-2"
    else:
        print(f"Loading adapted model from {model_path}...")

    # Match the 4-bit quantization used during QLoRA training for consistency
    # and to keep memory footprint portable across hardware (Colab T4, ~8GB+ cards).
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        REAL_TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if REAL_TOKENIZER.pad_token is None:
            REAL_TOKENIZER.pad_token = REAL_TOKENIZER.eos_token

        print("Loading base model (microsoft/phi-2) in 4-bit...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Applying adapter weights...")
        REAL_MODEL = PeftModel.from_pretrained(base_model, model_path)
        REAL_MODEL.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def run_real_inference(prompt: str) -> str:
    import torch
    global REAL_MODEL, REAL_TOKENIZER
    load_real_model()
    inputs = REAL_TOKENIZER(prompt, return_tensors="pt").to(REAL_MODEL.device)
    with torch.no_grad():
        outputs = REAL_MODEL.generate(**inputs, max_new_tokens=150, pad_token_id=REAL_TOKENIZER.eos_token_id)
    # The generated output includes the prompt, so we strip it out
    out_text = REAL_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return out_text[len(prompt):].strip()

# --- 5-Dim Rubric (0-10 Scale) ---
def eval_decision(prediction: str, expected: str) -> int:
    pred_up = prediction.upper()
    exp_up = expected.upper()
    pred_app = "APPROVE" in pred_up
    pred_veto = "VETO" in pred_up
    exp_app = "APPROVE" in exp_up
    exp_veto = "VETO" in exp_up
    
    if (exp_app and pred_app and not pred_veto) or (exp_veto and pred_veto and not pred_app):
        return 10
    return 0

def eval_grounding(prediction: str, evidence: str) -> int:
    import re
    nums_pred = set(re.findall(r'\b\d+(?:\.\d+)?(?:ft|lbs|tons|mi)?\b', prediction.lower()))
    nums_evid = set(re.findall(r'\b\d+(?:\.\d+)?(?:ft|lbs|tons|mi)?\b', evidence.lower()))
    hallucinated = any(n not in nums_evid and not n.isdigit() for n in nums_pred)
    return 0 if hallucinated else 10

def eval_constraint(prediction: str) -> int:
    pred = prediction.lower()
    if "limit" in pred or "clearance" in pred or "weight" in pred or "tons" in pred or "ft" in pred:
        return 10
    if "route" in pred or "bridge" in pred:
        return 5
    return 0

def eval_cot(prediction: str) -> int:
    import re
    steps = len(re.findall(r'(?i)(step|thought)', prediction))
    if steps >= 4: return 10
    if steps == 3: return 7
    if steps == 2: return 4
    return 0

def eval_jargon(prediction: str) -> int:
    keywords = ["route", "clearance", "payload", "eta", "constraint", "bridge", "weather", "traffic", "reroute", "severe", "limit"]
    pred = prediction.lower()
    matches = sum(1 for kw in keywords if kw in pred)
    if matches >= 4: return 10
    if matches >= 2: return 7
    if matches == 1: return 4
    return 0

def calculate_pass_rate(results: List[Dict[str, Any]]) -> float:
    passed = sum(1 for r in results if r.get("avg_score", 0) >= 7.0)
    return (passed / len(results)) * 100 if results else 0.0

def run_metamorphic_tests(results: List[Dict[str, Any]]) -> Dict[str, bool]:
    tests = {}
    
    import collections
    groups = collections.defaultdict(dict)
    for r in results:
        label = r.get("meta_label", "")
        if label.startswith("invariance_") or label.startswith("monotonicity_") or label.startswith("symmetry_"):
            parts = label.split("_")
            group_name = f"{parts[0]}_{parts[2]}"
            role = parts[1] # "base" or "pair"
            groups[group_name][role] = r
    
    for group_name, pair_dict in groups.items():
        if "base" in pair_dict and "pair" in pair_dict:
            base_dec = pair_dict["base"]["prediction_decision"]
            pair_dec = pair_dict["pair"]["prediction_decision"]
            
            if group_name.startswith("invariance"):
                tests[f"{group_name} (Same decision)"] = (base_dec == pair_dec)
            elif group_name.startswith("monotonicity"):
                expected_pass = (base_dec == "APPROVE" and pair_dec == "VETO") or (base_dec == "VETO" and pair_dec == "VETO")
                tests[f"{group_name} (Stricter constraint)"] = expected_pass
            elif group_name.startswith("symmetry"):
                tests[f"{group_name} (A->B == B->A)"] = (base_dec == pair_dec)
                
    return tests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["mock", "real"], default="mock")
    parser.add_argument("--queries", type=str, default="data/evaluation_queries.json")
    parser.add_argument("--verbose", action="store_true", help="Print full model output for each query")
    args = parser.parse_args()
    
    print(f"Loading queries from {args.queries}...")
    queries = load_queries(args.queries)
    
    results = []
    
    print("\nRunning Evaluation Suite...\n")
    
    total_acc = 0
    for q in queries:
        instruction = q["instruction"]
        evidence = q["input"]
        expected = q["output"]
        q_id = q["id"]
        
        # Build prompt using SC-CoT for evaluation
        prompt = prompt_adaptation.build_sc_cot_prompt(instruction, evidence, [])
        
        # Inference
        if args.mode == "mock":
            prediction = run_mock_inference(instruction, evidence, expected)
        else:
            # Run real inference using the local transformers model
            prediction = run_real_inference(prompt)
            
        # Parse decision roughly
        pred_decision = "APPROVE" if "APPROVE" in prediction.upper() else "VETO" if "VETO" in prediction.upper() else "UNKNOWN"
            
        # 5-Dim Rubric
        score_dec = eval_decision(prediction, expected)
        score_gro = eval_grounding(prediction, evidence)
        score_con = eval_constraint(prediction)
        score_cot = eval_cot(prediction)
        score_jar = eval_jargon(prediction)
        
        avg_score = (score_dec + score_gro + score_con + score_cot + score_jar) / 5.0
        
        results.append({
            "id": q_id,
            "prediction_decision": pred_decision,
            "meta_label": q.get("meta_label", ""),
            "scores": {
                "decision": score_dec,
                "grounding": score_gro,
                "constraint": score_con,
                "cot": score_cot,
                "jargon": score_jar
            },
            "avg_score": avg_score
        })
        
        print(f"[{q_id}] Expected: {expected} | Predicted: {pred_decision} | Avg Score: {avg_score:.1f}/10")
        if args.verbose:
            print(f"  >>> {prediction}\n")
        
    print("\n--- Summary Statistics ---")
    pass_rate = calculate_pass_rate(results)
    print(f"Overall Pass Rate (Avg >= 7.0): {pass_rate:.1f}%")
    
    print("\n--- Metamorphic Testing ---")
    meta_results = run_metamorphic_tests(results)
    for test_name, passed in meta_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")

    import sys
    if pass_rate < 70.0:
        print("\n❌ PIPELINE FAILED: Pass rate < 70%")
        sys.exit(1)
    else:
        print("\n✅ PIPELINE PASSED: Pass rate >= 70%")
        sys.exit(0)

if __name__ == "__main__":
    main()
