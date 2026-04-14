"""
HFACS Batch Extractor for NTSB Data — Ollama (local, free, no limits)
=======================================================================
Reads ntsb_text_fields.csv, uses a local Ollama model to classify each
row into HFACS features, and writes results to hfacs_results.csv.

Setup:
    1. Install Ollama: https://ollama.com
    2. Pull a model:
          ollama pull qwen2.5:7b      (good for 8GB RAM)
          ollama pull llama3.1:8b     (better, needs 16GB RAM)
    3. Install Python deps:
          pip install ollama pandas tqdm
    4. Make sure Ollama is running (it starts automatically on install)
    5. Run:
          python hfacs_extractor_ollama.py

Optional flags:
    python hfacs_extractor_ollama.py --model llama3.1:8b
    python hfacs_extractor_ollama.py --limit 10        # test on 10 rows
    python hfacs_extractor_ollama.py --resume          # continue after interruption
    python hfacs_extractor_ollama.py --input my_data.csv --output results.csv

Speed estimate:
    qwen2.5:7b  ~ 3-8 sec/row  -> 1,685 rows ~ 2-4 hours
    llama3.1:8b ~ 5-15 sec/row -> 1,685 rows ~ 3-7 hours
    (runs faster with a GPU, slower on CPU-only)
"""

import json
import math
import time
import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import ollama

# ---------------------------------------------------------------------------
# Config — change model here if needed
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "qwen2.5:7b"   # or "llama3.1:8b"

# ---------------------------------------------------------------------------
# HFACS schema
# ---------------------------------------------------------------------------

HFACS_SCHEMA = {
    "org_climate": {
        "label": "Organizational Climate",
        "subs": ["Safety Climate", "Culture", "Structure"],
    },
    "supervisory": {
        "label": "Supervisory Conditions",
        "subs": [
            "Inadequate Supervision",
            "Planned Inappropriate Operations",
            "Failed to Correct Known Problem",
            "Supervisory Violations",
        ],
    },
    "personnel": {
        "label": "Personnel Conditions",
        "subs": ["Crew Resource Management", "Personal Readiness"],
    },
    "operator": {
        "label": "Operator Conditions",
        "subs": [
            "Adverse Mental State",
            "Adverse Physiological State",
            "Physical Limitations",
            "Mental Limitations",
        ],
    },
    "unsafe": {
        "label": "Unsafe Conditions",
        "subs": [
            "Decision Errors",
            "Skill-based Errors",
            "Perceptual Errors",
            "Routine Violations",
            "Exceptional Violations",
        ],
    },
}

OUTPUT_COLS = []
for cat_id, cat in HFACS_SCHEMA.items():
    OUTPUT_COLS.append(cat["label"])
    for sub in cat["subs"]:
        OUTPUT_COLS.append(sub)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an HFACS (Human Factors Analysis and Classification System) expert "
    "analyzing aviation accident data. You respond ONLY with valid JSON. "
    "Never include markdown, code fences, or any explanation outside the JSON."
)

MAX_FIELD_LEN = 800  # chars per field — keeps total prompt within 7B model context

def _trim(text, max_len=MAX_FIELD_LEN):
    if text is None:
        return "(not provided)"
    if isinstance(text, float) and math.isnan(text):
        return "(not provided)"
    text = str(text).strip()
    if not text or text.lower() == "nan":
        return "(not provided)"
    return text[:max_len] + "…" if len(text) > max_len else text


def build_prompt(probable_cause, findings, narrative, recommendations, force_binary=False):
    if force_binary:
        scale = "YES or NO only — make your best judgment from the available text, never use UNKNOWN"
        scale_hint = "  YES = indicated or likely based on context\n  NO  = absent or not indicated"
    else:
        scale = "YES, NO, or UNKNOWN"
        scale_hint = "  YES     = clearly indicated by the text\n  NO      = text indicates absence or irrelevance\n  UNKNOWN = not enough information"
    return f"""Analyze this NTSB accident text and classify HFACS features.

For each category and subcategory use ONLY {scale}
{scale_hint}

TEXT:
PROBABLE CAUSE: {_trim(probable_cause)}
FINDINGS: {_trim(findings)}
NARRATIVE: {_trim(narrative)}
RECOMMENDATIONS: {_trim(recommendations)}

Respond with ONLY this JSON (no other text):
{{
  "org_climate": {{
    "present": "YES|NO|UNKNOWN",
    "subs": {{
      "Safety Climate": {{"present": "YES|NO|UNKNOWN"}},
      "Culture": {{"present": "YES|NO|UNKNOWN"}},
      "Structure": {{"present": "YES|NO|UNKNOWN"}}
    }}
  }},
  "supervisory": {{
    "present": "YES|NO|UNKNOWN",
    "subs": {{
      "Inadequate Supervision": {{"present": "YES|NO|UNKNOWN"}},
      "Planned Inappropriate Operations": {{"present": "YES|NO|UNKNOWN"}},
      "Failed to Correct Known Problem": {{"present": "YES|NO|UNKNOWN"}},
      "Supervisory Violations": {{"present": "YES|NO|UNKNOWN"}}
    }}
  }},
  "personnel": {{
    "present": "YES|NO|UNKNOWN",
    "subs": {{
      "Crew Resource Management": {{"present": "YES|NO|UNKNOWN"}},
      "Personal Readiness": {{"present": "YES|NO|UNKNOWN"}}
    }}
  }},
  "operator": {{
    "present": "YES|NO|UNKNOWN",
    "subs": {{
      "Adverse Mental State": {{"present": "YES|NO|UNKNOWN"}},
      "Adverse Physiological State": {{"present": "YES|NO|UNKNOWN"}},
      "Physical Limitations": {{"present": "YES|NO|UNKNOWN"}},
      "Mental Limitations": {{"present": "YES|NO|UNKNOWN"}}
    }}
  }},
  "unsafe": {{
    "present": "YES|NO|UNKNOWN",
    "subs": {{
      "Decision Errors": {{"present": "YES|NO|UNKNOWN"}},
      "Skill-based Errors": {{"present": "YES|NO|UNKNOWN"}},
      "Perceptual Errors": {{"present": "YES|NO|UNKNOWN"}},
      "Routine Violations": {{"present": "YES|NO|UNKNOWN"}},
      "Exceptional Violations": {{"present": "YES|NO|UNKNOWN"}}
    }}
  }}
}}"""

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def parse_response(raw: str) -> dict:
    """Strip any markdown fences and parse JSON."""
    clean = raw.strip()
    # Remove markdown code fences if model adds them despite instructions
    if "```" in clean:
        parts = clean.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                clean = part
                break
    # Find the JSON object if there's extra text around it
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start != -1 and end > start:
        clean = clean[start:end]
    return json.loads(clean)


def flatten_result(parsed: dict) -> dict:
    row = {}
    for cat_id, cat in HFACS_SCHEMA.items():
        cat_data = parsed.get(cat_id, {})
        row[cat["label"]] = cat_data.get("present", "UNKNOWN")
        for sub in cat["subs"]:
            sub_data = cat_data.get("subs", {}).get(sub, {})
            row[sub] = sub_data.get("present", "UNKNOWN")
    return row


def extract_row(model_name: str, row: pd.Series, retries: int = 3, force_binary: bool = False) -> dict:
    prompt = build_prompt(
        probable_cause=row.get("ProbableCause"),
        findings=row.get("Findings"),
        narrative=row.get("narratives.narr_cause"),
        recommendations=row.get("safetyrecs2026-02-28_15-34.Recommendation"),
        force_binary=force_binary,
    )

    for attempt in range(1, retries + 1):
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                options={"temperature": 0.0},  # deterministic output
            )
            raw = response["message"]["content"]
            parsed = parse_response(raw)
            return flatten_result(parsed)

        except json.JSONDecodeError as e:
            logging.warning(f"JSON parse error attempt {attempt}: {e}")
            logging.debug(f"Raw response was: {repr(raw[:300])}")
            if attempt < retries:
                time.sleep(1)
                continue

        except Exception as e:
            err = str(e)
            if "connection" in err.lower() or "refused" in err.lower():
                raise SystemExit(
                    "\nERROR: Cannot connect to Ollama.\n"
                    "Make sure Ollama is running — open the Ollama app or run 'ollama serve' in a terminal."
                )
            logging.error(f"Error on attempt {attempt}: {e}")
            if attempt < retries:
                time.sleep(2)
                continue

    logging.warning(f"All retries failed for row {row.get('NtsbNo')} — marking as ERROR")
    return {col: "ERROR" for col in OUTPUT_COLS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HFACS extractor using local Ollama model")
    _here = Path(__file__).parent
    parser.add_argument("--input",   default=str(_here / "ntsb text fields.csv"), help="Input CSV path")
    parser.add_argument("--output",  default=str(_here / "hfacs_results.csv"),    help="Output CSV path")
    parser.add_argument("--model",          default=DEFAULT_MODEL,  help="Ollama model name")
    parser.add_argument("--limit",          type=int, default=None, help="Process only first N rows")
    parser.add_argument("--resume",         action="store_true",    help="Skip already-processed rows")
    parser.add_argument("--rerun-errors",   action="store_true",    help="Re-extract rows marked ERROR in existing output")
    parser.add_argument("--rerun-unknowns", action="store_true",    help="Re-extract rows with any UNKNOWN values in existing output")
    parser.add_argument("--force-binary",   action="store_true",    help="Force YES/NO only — no UNKNOWN allowed (best for ML)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Verify Ollama is reachable and model is available
    try:
        models = [m.model for m in ollama.list().models]
        logging.info(f"Ollama running. Available models: {models}")
        # Check if requested model is pulled
        if not any(args.model in m for m in models):
            raise SystemExit(
                f"\nModel '{args.model}' not found locally.\n"
                f"Pull it first by running:\n"
                f"    ollama pull {args.model}\n"
                f"Available models: {models}"
            )
    except Exception as e:
        if "SystemExit" in type(e).__name__:
            raise
        raise SystemExit(
            "\nERROR: Cannot connect to Ollama.\n"
            "Make sure Ollama is running — open the Ollama app or run 'ollama serve' in a terminal."
        )

    logging.info(f"Using model: {args.model}")

    # Load input
    df = pd.read_csv(args.input, encoding="latin1")
    if args.limit:
        df = df.head(args.limit)
    logging.info(f"Loaded {len(df)} rows from {args.input}")

    # Determine which rows to skip based on mode
    already_done = set()
    out_path = Path(args.output)
    if out_path.exists():
        done_df = pd.read_csv(out_path, encoding="latin1")
        hfacs_cols = [c for c in done_df.columns if c not in ["NtsbNo", "Event.Id"]]
        if args.rerun_errors:
            clean = done_df[~done_df[hfacs_cols].eq("ERROR").any(axis=1)]
            already_done = set(clean["NtsbNo"].astype(str).tolist())
            logging.info(f"Rerun-errors: skipping {len(already_done)} clean rows, rerunning {len(done_df) - len(already_done)} ERROR rows")
        elif args.rerun_unknowns:
            clean = done_df[~done_df[hfacs_cols].isin(["ERROR", "UNKNOWN"]).any(axis=1)]
            already_done = set(clean["NtsbNo"].astype(str).tolist())
            logging.info(f"Rerun-unknowns: skipping {len(already_done)} clean rows, rerunning {len(done_df) - len(already_done)} rows")
        elif args.resume:
            already_done = set(done_df["NtsbNo"].astype(str).tolist())
            logging.info(f"Resuming: {len(already_done)} rows already processed")

    rows_to_process = df[~df["NtsbNo"].astype(str).isin(already_done)]
    logging.info(f"Rows to process: {len(rows_to_process)}")

    if rows_to_process.empty:
        logging.info("All rows already processed.")
        return

    result_rows = []

    for _, row in tqdm(rows_to_process.iterrows(), total=len(rows_to_process), desc=f"Extracting [{args.model}]"):
        ntsb_no = str(row["NtsbNo"])
        features = extract_row(args.model, row, force_binary=args.force_binary)
        base = {"NtsbNo": ntsb_no, "Event.Id": row.get("Event.Id", "")}
        base.update(features)
        result_rows.append(base)

        # Save checkpoint every 25 rows
        if len(result_rows) % 25 == 0:
            _save(result_rows, args.output, resume=True, out_path=out_path)
            logging.info(f"Checkpoint saved ({len(result_rows)} new rows processed)")

    # Final save
    _save(result_rows, args.output, resume=True, out_path=out_path)
    logging.info(f"Done. Results saved to {args.output}")

    # Summary
    result_df = pd.read_csv(args.output, encoding="latin1")
    print(f"\n--- HFACS Presence Summary ({len(result_df)} total rows) ---")
    for col in OUTPUT_COLS:
        if col in result_df.columns:
            counts = result_df[col].value_counts()
            yes = counts.get("YES", 0)
            unk = counts.get("UNKNOWN", 0)
            total = len(result_df)
            print(f"  {col:<42} YES={yes:4d} ({100*yes/total:.0f}%)  UNK={unk:4d} ({100*unk/total:.0f}%)")


def _save(result_rows, output_path, resume, out_path):
    new_df = pd.DataFrame(result_rows)
    if resume and out_path.exists():
        existing = pd.read_csv(output_path, encoding="latin1")
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["NtsbNo"], keep="last", inplace=True)
        combined.to_csv(output_path, index=False)
    else:
        new_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()