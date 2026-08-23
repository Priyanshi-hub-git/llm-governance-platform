"""
evaluator.py

Core black-box evaluation logic for Layer 1.
- get_model_response(): sends ONE prompt to the model via litellm, returns
  (response_text, cost_in_usd)
- run_full_evaluation(): loops through every prompt in TEST_PROMPTS,
  scores each response via scoring.py, and collects everything + running
  total cost into a Pandas DataFrame

Design note: this file doesn't know or care which provider (OpenAI, Gemini,
Anthropic, etc.) is behind model_id -- litellm handles that. This file only
deals with "send prompt, get text + cost back, score it."
"""

import litellm
litellm.suppress_debug_info = True

import pandas as pd
from litellm import completion, completion_cost

from layer1.prompts import TEST_PROMPTS
from layer1.scoring import score_response


def get_model_response(model_id, api_key, prompt):
    """
    Sends a single prompt to the given model via litellm.
    Returns (response_text, cost_in_usd).
    On failure, returns (error_message_string, 0.0) -- caller checks for
    the "ERROR:" prefix rather than relying on exceptions bubbling up,
    so one bad call doesn't crash the whole evaluation loop.
    """
    try:
        response = completion(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
        )
        text = response.choices[0].message.content

        try:
            cost = completion_cost(completion_response=response)
        except Exception:
            # Some models (very new / obscure, or free-tier) aren't in
            # litellm's pricing map -- don't let a missing price break
            # the evaluation, just report $0.
            cost = 0.0

        return text, cost

    except Exception as e:
        return f"ERROR: {str(e)}", 0.0


def test_connection(model_id, api_key):
    """
    Cheap single call used to validate model_id + api_key BEFORE running
    the full prompt battery. Saves the user from waiting through 6+ calls
    only to find out the key or model name was wrong on the first one.
    Returns (is_valid: bool, message: str)
    """
    text, _ = get_model_response(model_id, api_key, "Hello")
    if text.startswith("ERROR:"):
        return False, text
    return True, "Connection successful."


def run_full_evaluation(model_id, api_key):
    """
    Runs every prompt in TEST_PROMPTS against the model, scoring each
    response as it comes back (via scoring.score_response).

    Returns (results_dataframe, total_cost_usd)

    results_dataframe columns:
        category, prompt, response, cost_usd, risk_score
        (risk_score: 0.0 = no risk/good, 1.0 = high risk/bad -- see
        scoring.py's module docstring for why this direction was chosen)
    """
    results = []
    total_cost = 0.0

    for category, prompts in TEST_PROMPTS.items():
        for prompt in prompts:
            response_text, cost = get_model_response(model_id, api_key, prompt)
            total_cost += cost

            results.append({
                "category": category,
                "prompt": prompt,
                "response": response_text,
                "cost_usd": cost,
                "risk_score": score_response(category, response_text),
            })

    df = pd.DataFrame(results)
    return df, total_cost


if __name__ == "__main__":
    # Manual smoke test -- run this file directly:
    # python layer1/evaluator.py
    #
    # NOTE: this will make REAL API calls (tiny/free cost depending on the
    # model you use). Replace the placeholders below with a real model_id
    # and API key before running.

    TEST_MODEL_ID = "gpt-4o-mini"       # any valid litellm model_id
    TEST_API_KEY = "sk-REPLACE-ME"       # your real key -- never commit this

    if TEST_API_KEY == "sk-REPLACE-ME":
        print("Set TEST_API_KEY to a real key before running this file directly.")
    else:
        ok, msg = test_connection(TEST_MODEL_ID, TEST_API_KEY)
        print(f"Connection check: {ok} -- {msg}")

        if ok:
            df, total_cost = run_full_evaluation(TEST_MODEL_ID, TEST_API_KEY)

            for _, row in df.iterrows():
                print("=" * 60)
                print("Category :", row["category"])
                print("Prompt   :", row["prompt"])
                print("Risk     :", row["risk_score"])
                print("Cost     :", row["cost_usd"])
                print("Response :")
                print(row["response"])
                print()

            print("=" * 60)
            print(f"Total cost: ${total_cost:.6f}")