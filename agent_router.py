# agent_router.py
import json, re
from typing import Dict, Any, Tuple, List, Optional

from helper import route_case, THR_MAHA, FEATURE_IDX, rexnet_predict_prob, single_image_score
from TTAhelper import get_tta_stats
from MOEhelper import run_committee, moe_decisive  
from VLMhelper import vlm_diag_line
import os, json, re, time
import MOEhelper as MOE         

def _run_moe_only(state: Dict[str, Any]) -> None:
    # DEBUG: confirm which checkpoints MoE is using right now
    try:
        print("[agent_router] Active MoE specs:", getattr(MOE, "_COMMITTEE_SPECS", None), flush=True)
        if "fold" in state:
            print(f"[agent_router] Current fold: {state['fold']}", flush=True)
    except Exception as e:
        print("[agent_router] Could not inspect MoE specs:", repr(e), flush=True)
        
    moe = run_committee(state["image_path"])
    state["stage"] = "moe"
    state["moe"] = moe
    # decide if committee alone is enough
    decisive = moe_decisive(
        moe,
        min_support=None,      # or your thresholds
        min_total=None,
        min_conf=None,
    )
    if decisive:
        state["final_label"] = moe.get("label")
        state["decided_by"]  = "moe"

def _run_vlm_only(state: Dict[str, Any], vlm_model: str = "gpt-4.1-mini") -> None:
    vlm = vlm_diag_line(state["image_path"], model=vlm_model)
    state["stage"] = "vlm"
    state["vlm"] = vlm
    if vlm.get("ok"):
        pe_yes = (vlm["label"] == 1)
        state["final_label"] = "PE_yes" if pe_yes else "PE_no"
        state["p"] = vlm.get("confidence")  # keep prob in state for logging
        state["decided_by"] = "vlm"

def _safe_json_from_text(txt: str) -> dict | None:
    # Try straight parse
    try:
        return json.loads(txt)
    except Exception:
        pass
    # Try to pull the first {...} block
    m = re.search(r'\{.*\}', txt, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def _heuristic_fallback() -> Dict[str, Any]:
    # A tiny, deterministic plan if no LLM (or parsing fails)
    return {
        "next_tool": "tta",
        "reason": "no-LLM fallback: run TTA first for agreement",
        "stop": False,
        "final_label": None,
        "decided_by": None,
    }

def _call_llm_router(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    Return a dict like:
      {"next_tool": "tta|accept|moe|vlm|escalate_human",
       "reason": "...",
       "stop": false,
       "final_label": null,
       "decided_by": null}
    """
    with open("APIconfig.json") as f:
        config = json.load(f)

    api_key = config["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key

    # If no key, use heuristic
    if not os.getenv("OPENAI_API_KEY"):
        return _heuristic_fallback()

    try:
        # Import here to avoid hard dependency if user doesn't use LLM
        import openai
        client = openai

        # Very explicit JSON instruction + a short rubric
        rubric = (
            "Pick exactly ONE next_tool from {accept, tta, moe, vlm}.\n"
            "- Prefer TTA if it hasn't run and case is uncertain.\n"
            "- Prefer MoE if OOD + high_conf but no agreement.\n"
            "- Prefer VLM if low_conf (in/out of domain) or after MoE undecided.\n"
            "- Only ACCEPT if high_conf AND (in_domain OR TTA agrees).\n"
            "Return ONLY a JSON object with keys: next_tool, reason, stop, final_label, decided_by."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rubric + "\n\n" + user_prompt},
        ]

        # Up to 3 tries for clean JSON
        last_txt = None
        for _ in range(3):
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",   # keep consistent with your VLM choice
                messages=messages,
                temperature=0.0,
                max_tokens=200,
            )
            # print("\nLLM router response",resp)
            last_txt = resp.choices[0].message.content.strip()
            obj = _safe_json_from_text(last_txt)
            # print(obj)
            if isinstance(obj, dict) and "next_tool" in obj:
                # Normalize output
                obj.setdefault("reason", "")
                obj.setdefault("stop", False)
                obj.setdefault("final_label", None)
                obj.setdefault("decided_by", None)
                return obj

            # Nudge the model to correct format
            messages.append({"role": "assistant", "content": last_txt})
            messages.append({"role": "user", "content": "FORMAT ERROR. Return ONLY a JSON object with the required keys."})
            time.sleep(0.5)

        # If still failing: heuristic
        return _heuristic_fallback()

    except Exception:
        # Any API/parse error → heuristic
        return _heuristic_fallback()

# ====== 1) Prompts ==========================================================
_SYSTEM = """You are a safety-first controller for pulmonary edema triage.
Goal: reach a final label (PE_yes/PE_no) with minimal risk, cost, and latency.
Available tools: accept, tta, moe, vlm, escalate_human.
HARD RULES (never violate; suggestions may be overridden locally):
- Never auto-accept extreme OOD or very small p-values.
- Only accept when high confidence and either in-domain or TTA agrees.
- If OOD and high confidence without agreement -> prefer MoE; if MoE undecided -> VLM.
Return a pure JSON object with keys: next_tool, reason, stop, final_label, decided_by.
"""

def _build_user_prompt(state: Dict[str, Any]) -> str:
    tried = {"tta": bool(state.get("tta")), "moe": bool(state.get("moe")), "vlm": bool(state.get("vlm"))}
    available = [t for t, done in {**tried, "accept": False}.items() if not done]
    # Give the LLM compact state. You can add budget/latency later.
    s = {
        "p": state.get("p"),
        "frd_maha": state.get("frd_maha"),
        "thr_maha": state.get("thr_maha"),
        "tta_std": (state.get("tta") or {}).get("p_std"),
        "in_domain": state.get("in_domain"),
        "policy": state.get("policy"),
        "already_ran": tried,
        "available_tools": available,
    }
    # s = {
    #     "p": state.get("p"),
    #     "frd_maha": state.get("frd_maha"),
    #     "thr_maha": state.get("thr_maha"),
    #     "tta_std": (state.get("tta") or {}).get("p_std"),
    #     "in_domain": state.get("in_domain"),
    #     "policy": state.get("policy"),
    #     "already_ran": {
    #         "tta": bool(state.get("tta")),
    #         "moe": bool(state.get("moe")),
    #         "vlm": bool(state.get("vlm")),
    #     }
    # }
    return f"STATE:\n{s}\nDecide next_tool."

# ====== 2) Local guardrails via your route_case =============================
# NOTE "Loosen" Design 
def _guard_and_map_plan(state: Dict[str, Any], plan: Dict[str, Any], multiplier: float = 1.5) -> Dict[str, Any]:
    """
    Let the LLM choose freely among tools not yet used.
    Only safety rule kept: accept is allowed only when high_conf AND (in_domain OR agree).
    """
    mapping = {
        "accept": "ACCEPT",
        "tta":    "TTA",
        "moe":    "RUN_MOE",
        "vlm":    "ESCALATE_VLM",
    }

    pol = state["policy"]
    p   = float(state["p"])
    d   = float(state["frd_maha"])
    # thr = float(state["thr_maha"]) * float(multiplier)
    thr = float(state["thr_maha"]) 
    tried = {
        "tta": bool(state.get("tta")),
        "moe": bool(state.get("moe")),
        "vlm": bool(state.get("vlm")),
    }
    tta_std = (state.get("tta") or {}).get("p_std")

    # --- mode-adjusted thresholds (clamped) ---
    def clamp(x, lo, hi): return max(lo, min(hi, x))
    if pol["mode"] == "conservative":
        p_accept_adj = pol["p_accept"] + 0.20
        std_ok_adj   = pol["std_ok"] - 0.01
    elif pol["mode"] == "sensitive":
        p_accept_adj = pol["p_accept"] + 0.05
        std_ok_adj   = pol["std_ok"] + 0.02
    else:
        p_accept_adj = pol["p_accept"] + 0.10
        std_ok_adj   = pol["std_ok"]
    p_accept_adj = clamp(p_accept_adj, 0.50, 0.98)
    std_ok_adj   = clamp(std_ok_adj,   0.01, 0.20)

    # --- derived flags ---
    in_domain = d <= thr
    high_conf = (p >= p_accept_adj) or (p <= (1.0 - p_accept_adj))
    agree     = (tta_std is not None) and (tta_std <= std_ok_adj)

    allowed: set[str] = set()

    # Only safety guard for accept
    if high_conf and (in_domain or agree):
        allowed.add("accept")

    # Make all other tools choosable once
    for tool in ("tta", "moe", "vlm"):
        if not tried[tool]:
            allowed.add(tool)

    # If somehow nothing is allowed, pick a safe default
    if not allowed:
        allowed = {"vlm"}
    # print("\n",allowed)
    # Map LLM suggestion into allowed set
    llm_next = (plan.get("next_tool") or "").lower()
    if llm_next in allowed:
        return {"action": mapping[llm_next]}

    # Fallback preference (cheap → costly)
    for choice in ("accept", "tta", "moe", "vlm"):
        if choice in allowed:
            return {"action": mapping[choice]}

    return {"action": "ESCALATE_VLM"}


# NOTE "Tight" Design 
# def _guard_and_map_plan(state: Dict[str, Any], plan: Dict[str, Any], multiplier: float = 1.5) -> Dict[str, Any]:
#     """
#     Build a *set* of safe actions; let the LLM choose within it.
#     If its choice is unsafe/unavailable, fall back to a deterministic preference.
#     """
#     pol = state["policy"]
#     p   = float(state["p"])
#     d   = float(state["frd_maha"])
#     thr = float(state["thr_maha"]) * float(multiplier)

#     tried = {
#         "tta": bool(state.get("tta")),
#         "moe": bool(state.get("moe")),
#         "vlm": bool(state.get("vlm")),
#     }
#     tta_std = (state.get("tta") or {}).get("p_std")

#     # ---- mode-adjusted thresholds with clamps ----
#     def clamp(x, lo, hi): return max(lo, min(hi, x))
#     if pol["mode"] == "conservative":
#         p_accept_adj = pol["p_accept"] + 0.20
#         std_ok_adj   = pol["std_ok"] - 0.01
#     elif pol["mode"] == "sensitive":
#         p_accept_adj = pol["p_accept"] + 0.05
#         std_ok_adj   = pol["std_ok"] + 0.02
#     else:
#         p_accept_adj = pol["p_accept"] + 0.10
#         std_ok_adj   = pol["std_ok"]

#     p_accept_adj = clamp(p_accept_adj, 0.50, 0.98)
#     std_ok_adj   = clamp(std_ok_adj,   0.01, 0.20)

#     # ---- derived flags ----
#     in_domain = d <= thr
#     high_conf = (p >= p_accept_adj) or (p <= (1.0 - p_accept_adj))
#     agree     = (tta_std is not None) and (tta_std <= std_ok_adj)

#     # very OOD → never auto-accept (even if TTA agrees)
#     ood_mult  = d / max(thr, 1e-8)
#     # TODO very_ood detection disbaled
#     very_ood = False
#     # very_ood  = ood_mult > pol.get("max_auto_accept_frd_multiple", 2.0)

#     allowed: set[str] = set()

#     # Accept only when truly safe
#     if high_conf and (in_domain or agree) and not very_ood:
#         allowed.add("accept")

#     # TTA: only if not already run and it’s potentially useful
#     # OOD High Conf OR In_Domain and low conf
#     if (not tried["tta"]) and ((not in_domain and high_conf) or (in_domain and not high_conf) or (tta_std is None)):
#         allowed.add("tta")

#     # MoE: OOD + high conf but no agreement, and not already run
#     if (not in_domain) and high_conf and (not agree) and (not tried["moe"]):
#         allowed.add("moe")

#     # VLM: OOD + low conf, or in-domain + low conf without agreement, and not already run
#     if (((not in_domain) and (not high_conf)) or (in_domain and (not high_conf) and (not agree))) and (not tried["vlm"]):
#         allowed.add("vlm")

#     # If nothing is allowed, pick the safest escalation that we haven't tried yet
#     if not allowed:
#         if not tried["vlm"]:
#             allowed = {"vlm"}
#         elif not tried["moe"]:
#             allowed = {"moe"}
#         elif not tried["tta"]:
#             allowed = {"tta"}
#         else:
#             allowed = {"accept"}  # last resort; guardrails above already block unsafe accept

#     # --- map LLM plan into the safety envelope ---
#     llm_next = (plan.get("next_tool") or "").lower()
#     if llm_next in allowed:
#         return {"action": llm_next}

#     # Fallback preference inside the safe set (cheap → costly)
#     for choice in ("accept", "tta", "moe", "vlm"):
#         if choice in allowed:
#             return {"action": choice}

#     return {"action": "vlm"}



# ====== 3) Tool runners (wrap your existing funcs) =========================
def _run_tta(state: Dict[str, Any]) -> None:
    tta = get_tta_stats(state["image_path"], n=8, allow_hflip=False)
    state["tta"] = {"p_mean": tta["p_mean"], "p_std": tta["p_std"]}
    # Use TTA mean going forward
    state["p"] = float(tta["p_mean"])
    state["stage"] = "tta"

def _run_moe_or_vlm(state: Dict[str, Any], direct_vlm: bool = False) -> None:
    if direct_vlm:
        result, stage = cascade_moe_vlm(
            state["image_path"],
            run_committee_fn=lambda _: {"label": "undetermined", "ok": False},
            vlm_fn=vlm_diag_line,
            do_prints=True,
        )
    else:
        result, stage = cascade_moe_vlm(
            state["image_path"],
            run_committee_fn=run_committee,
            vlm_fn=vlm_diag_line,
            do_prints=True,
        )
    state["stage"] = stage
    state["moe"] = result if stage == "moe" else None
    state["vlm"] = result if stage == "vlm" else None
    if result and result.get("label") in {"PE_yes", "PE_no"}:
        state["final_label"] = result["label"]
        state["decided_by"] = "vlm" if stage == "vlm" else "moe"

# ====== 4) Public entry: one-call router loop ==============================
def run_router_loop(image_path: str, policy: Dict[str, Any], max_steps: int = 4
                   ) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    """
    Returns: (result, stage, trace)
      - result: {"label": "PE_yes/PE_no", "pe_prob": p, ...} or {"label": "undetermined"}
      - stage:  "base" | "tta" | "moe" | "vlm"
      - trace:  list of planning/execution steps for logging/debug
    """
    # Initial perception (keep exactly as you do now)
    THR_multiplier = 1.5
    p = rexnet_predict_prob(image_path)
    frd = single_image_score(image_path)
    d = float(frd["Mahalanobis"])

    state: Dict[str, Any] = {
        "image_path": image_path,
        "policy": policy,
        "p": float(p),
        "frd_maha": d,
        "thr_maha": float(THR_MAHA * THR_multiplier),
        "in_domain": bool(d <= THR_MAHA * THR_multiplier),
        "stage": "base",
        "tta": None, "moe": None, "vlm": None,
        "final_label": None, "decided_by": None,
    }
    trace: List[Dict[str, Any]] = []
    # Quick accept via guardrails (optional): we still let LLM opine, but this speeds easy cases.
    rc = route_case(
        frd_maha=d, thr_maha=THR_MAHA, p=state["p"],
        p_accept=policy["p_accept"], mode=policy["mode"],
        ensemble_std=None, std_ok=policy["std_ok"],
        df_features=len(FEATURE_IDX),
        # use_pvalue_gate=policy["use_pvalue_gate"],
        # pvalue_min=policy["pvalue_min"],
        max_auto_accept_frd_multiple=policy["max_auto_accept_frd_multiple"],
        may_request_tta=True,
    )
    if rc["action"] == "ACCEPT":
        label = "PE_yes" if state["p"] >= policy["p_accept"] else "PE_no"
        result = {"label": label, "pe_prob": state["p"]}
        trace.append({"auto_accept": True, "rc": rc})
        return result, "base", trace

    # Plan–Act loop
    for step in range(max_steps):
        plan = _call_llm_router(_SYSTEM, _build_user_prompt(state))
        safe = _guard_and_map_plan(state, plan, THR_multiplier)
        trace.append({"plan": plan, "mapped": safe, "step": step})

        if safe["action"] == "TTA":
            _run_tta(state)
            continue

        if safe["action"] == "RUN_MOE":
            _run_moe_only(state)
            if state["final_label"]:
                break
            # MoE undecided → next iteration will likely map to VLM
            continue

        if safe["action"] == "ESCALATE_VLM":
            _run_vlm_only(state)
            break

        if safe["action"] == "ACCEPT":
            label = "PE_yes" if state["p"] >= policy["p_accept"] else "PE_no"
            state["final_label"] = label
            state["decided_by"] = "model" if state["stage"] == "base" else "model"
            break

    # Shape result like your current code expects
    if state["final_label"] in {"PE_yes", "PE_no"}:
        result = {
            "label": state["final_label"],
            "pe_prob": state["p"],
            "decided_by": state["decided_by"],   # <-- add this
        }
        return result, state["stage"], trace

    # fallback if nothing decided
    return {"label": "undetermined"}, state["stage"], trace
