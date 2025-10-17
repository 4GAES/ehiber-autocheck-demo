#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone evaluator para GitHub Actions (sin rubric-grader).

Features:
- Carga y fusiona "rubrics_chain" (module/stack/global) desde ./rubricas/
- Escaneo agnóstico: required_files/any_of/forbidden_globs + pick_rules -> submissions/
- Scoring A/B/C/D con OpenAI Chat Completions (por defecto: gpt-5-nano)
- Artefactos: results.csv, evaluation.log, scanner_meta.json, rubrica_efectiva.yaml, ISSUE_BODY.md

Requiere:
- OPENAI_API_KEY (env)
- (Opcional) PyYAML. Si no está, intenta auto-instalarlo (desactivable con NO_AUTO_PIP=1)
"""

import os, sys, re, glob, json, csv, time, subprocess, statistics, base64, shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# ---------- YAML loader (auto-instala PyYAML si hace falta) ----------
def _ensure_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except Exception:
        if os.getenv("NO_AUTO_PIP") == "1":
            raise
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "PyYAML", "-q"])
            import yaml  # type: ignore
            return yaml
        except Exception as e:
            raise RuntimeError("No se pudo cargar/instalar PyYAML. Usa rúbricas en JSON o habilita internet del runner.") from e

# ---------- Utilidades ----------
def log(msg: str):
    with open("evaluation.log", "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")
    print(msg)

def read_text(path: Path, max_bytes: Optional[int] = None) -> str:
    if not path.exists() or not path.is_file():
        return ""
    b = path.read_bytes()
    if max_bytes is not None:
        b = b[:max_bytes]
    return b.decode("utf-8", errors="ignore")

def parse_ref(s: str) -> Tuple[str, Optional[str]]:
    m = re.match(r"(.+?)@(.*)", s)
    return (m.group(1), m.group(2)) if m else (s, None)

def pick_version(base: str, spec: Optional[str]) -> Optional[str]:
    # base es algo como "modules/intro_web.yaml" o "modules/intro_web"
    base_no_yaml = base[:-5] if base.endswith(".yaml") else base
    if not spec:
        # elige la mayor "path@*.yaml" si existe, si no path.yaml
        cands = sorted(glob.glob(os.path.join("rubricas", base_no_yaml + "@*.yaml")))
        return cands[-1] if cands else os.path.join("rubricas", base_no_yaml + ".yaml")
    if spec.endswith(".yaml") and not any(ch in spec for ch in "<>="):
        return os.path.join("rubricas", f"{base_no_yaml}@{spec}")
    # rangos semver simplificados -> tomar la mayor disponible
    cands = sorted(glob.glob(os.path.join("rubricas", base_no_yaml + "@*.yaml")))
    return cands[-1] if cands else None

def deep_merge(a, b):
    # fusiona b "sobre" a (b completa/pisa según __op)
    if isinstance(a, dict) and isinstance(b, dict):
        if b.get("__op") == "replace":
            return {k: v for k, v in b.items() if k != "__op"}
        out = dict(a)
        for k, v in b.items():
            if k == "__op":
                continue
            out[k] = deep_merge(out[k], v) if k in out else v
        return out
    if isinstance(a, list) and isinstance(b, list):
        # listas de criterios -> merge por id
        if all(isinstance(x, dict) and "id" in x for x in a + b):
            idx = {x["id"]: x for x in a}
            for it in b:
                if it.get("__op") == "replace":
                    idx[it["id"]] = {k: v for k, v in it.items() if k != "__op"}
                elif it["id"] in idx:
                    idx[it["id"]] = deep_merge(idx[it["id"]], it)
                else:
                    idx[it["id"]] = it
            return list(idx.values())
        return a + b
    return a

# ---------- OpenAI (Chat Completions) ----------
def openai_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    import urllib.request, urllib.error
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY")
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    data = json.dumps(body).encode("utf-8")
    try:
        with urllib.request.urlopen(req, data, timeout=120) as resp:
            out = json.loads(resp.read().decode("utf-8", errors="ignore"))
            # Soportar formatos distintos
            if "choices" in out and out["choices"]:
                c = out["choices"][0]
                content = c.get("message", {}).get("content", "")
                return content
            return ""
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        log(f"[OpenAI HTTPError] {e.code} {err}")
        raise
    except Exception as e:
        log(f"[OpenAI ERROR] {e}")
        raise

def parse_json_from_text(txt: str) -> Dict[str, Any]:
    txt = txt.strip()
    # Intentar JSON directo
    try:
        return json.loads(txt)
    except Exception:
        pass
    # Intentar bloque ```json ... ```
    m = re.search(r"```json\s*(.*?)```", txt, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Último recurso: extraer llaves balanceadas
    start = txt.find("{")
    end = txt.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(txt[start:end + 1])
        except Exception:
            pass
    raise ValueError("No se pudo parsear JSON de la respuesta del modelo.")

# ---------- Prompts ----------
ONE_STEP_PROMPT = (
    "Eres un evaluador experto. Recibirás el enunciado del proyecto, una rúbrica y un conjunto de fragmentos del repositorio.\n"
    "Asigna puntajes por criterio (0..peso) y feedback breve. Responde SOLO JSON con:\n"
    "{\n"
    '  "criterios": [{"id":"<id>","score":int,"max":int,"feedback":[str,...]}],\n'
    '  "total": int\n'
    "}\n\n"
    "Enunciado:\n{problem}\n\nRúbrica (texto/criterios):\n{rubric}\n\nFragmentos:\n{snippets}\n"
)

TWO_STEP_1 = (
    "Eres un evaluador. Resume la intención lógica del código a alto nivel (arquitectura, flujo, componentes) en 10-15 viñetas.\n"
    "Código/fragmentos:\n{snippets}\n"
)
TWO_STEP_2 = (
    "Ahora, con el enunciado y la rúbrica, compara con la intención lógica y puntúa criterios.\n"
    "Responde SOLO JSON como en el formato indicado antes.\n\n"
    "Enunciado:\n{problem}\n\nRúbrica:\n{rubric}\n\nIntención inferida:\n{logic}\n"
)

AIO_PROMPT = (
    "Evaluación one-shot: puntúa según la rúbrica y el enunciado. Responde SOLO JSON (formato indicado antes).\n"
    "Enunciado:\n{problem}\n\nRúbrica:\n{rubric}\n\nFragmentos:\n{snippets}\n"
)

# ---------- Scanner ----------
def run_scanner(rubrica: Dict[str, Any]) -> Dict[str, Any]:
    struct = rubrica.get("estructura", {})
    required = struct.get("required_files", []) or []
    any_groups = struct.get("required_any_of", []) or []
    forbidden = struct.get("forbidden_globs", []) or []
    max_total = int(struct.get("max_context_bytes", 120_000))

    penalizaciones: List[str] = []
    for p in required:
        if not Path(p).exists():
            penalizaciones.append(f"missing:{p}")
    for group in any_groups:
        if not any(Path(x).exists() for x in group):
            penalizaciones.append(f"missing_any_of:{group}")
    for pat in forbidden:
        if glob.glob(pat, recursive=True):
            penalizaciones.append(f"forbidden:{pat}")

    rules = struct.get("pick_rules") or [
        {"globs": ["README.md"], "take": 1, "max_bytes": 20_000},
        {"globs": ["**/*.html", "**/*.tsx", "**/*.jsx"], "take": 2, "max_bytes": 30_000},
        {"globs": ["src/**/*.js", "src/**/*.ts", "src/**/*.py"], "take": 3, "max_bytes": 25_000},
        {"globs": ["**/*.css", "**/*.scss"], "take": 1, "max_bytes": 10_000},
        {"globs": ["**/*.test.*", "**/__tests__/**/*"], "take": 2, "max_bytes": 20_000},
    ]

    picked: List[Tuple[str, int]] = []
    used = 0
    def add_file(p: str, cap: int):
        nonlocal used
        if used >= max_total:
            return
        if not os.path.isfile(p):
            return
        size = os.path.getsize(p)
        take = min(size, cap, max_total - used)
        if take <= 0: return
        picked.append((p, take))
        used += take

    for rule in rules:
        takes = int(rule.get("take", 1))
        cap = int(rule.get("max_bytes", 20_000))
        matches: List[str] = []
        for g in rule.get("globs", []):
            matches += glob.glob(g, recursive=True)
        matches = sorted(set(matches), key=lambda x: os.path.getsize(x) if os.path.isfile(x) else 0, reverse=True)
        for m in matches[:takes]:
            add_file(m, cap)

    subdir = Path("submissions")
    subdir.mkdir(exist_ok=True)
    snippets = []
    for i, (src, take) in enumerate(picked, 1):
        ext = os.path.splitext(src)[1] or ".txt"
        dst = subdir / f"student_{i}{ext}"
        with open(src, "rb") as s, open(dst, "wb") as d:
            d.write(s.read(take))
        text = read_text(dst)
        snippets.append(f"--- {src} ---\n{text}\n")
    meta = {"picked_files": [p for p,_ in picked], "bytes": used, "penalties": penalizaciones}
    Path("scanner_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"snippets": "\n".join(snippets), "meta": meta}

# ---------- Evaluación ----------
def evaluate(scoring: Dict[str, Any], rubrica_text: str, problem_text: str, snippets: str) -> Dict[str, Any]:
    model = scoring.get("model", "gpt-5-nano")
    stype = scoring.get("type", "A")
    ensemble = int(scoring.get("ensemble_size", 1))

    def one_call(prompt: str) -> Dict[str, Any]:
        msg = [{"role": "system", "content": "Eres un evaluador que responde SOLO JSON válido."},
               {"role": "user", "content": prompt}]
        out = openai_chat(model, msg, temperature=0.1)
        return parse_json_from_text(out)

    if stype == "A":
        data = one_call(ONE_STEP_PROMPT.format(problem=problem_text, rubric=rubrica_text, snippets=snippets))
        return data

    if stype == "B":
        logic = openai_chat(model, [{"role":"system","content":"Eres un evaluador."},
                                    {"role":"user","content": TWO_STEP_1.format(snippets=snippets)}], temperature=0.0)
        data = one_call(TWO_STEP_2.format(problem=problem_text, rubric=rubrica_text, logic=logic))
        return data

    if stype == "C":
        runs: List[Dict[str, Any]] = []
        for i in range(ensemble):
            data = one_call(ONE_STEP_PROMPT.format(problem=problem_text, rubric=rubrica_text, snippets=snippets))
            runs.append(data)
        # consenso simple: promedio de scores por criterio id, y total promedio
        by_id: Dict[str, List[int]] = {}
        max_by_id: Dict[str, int] = {}
        for r in runs:
            for c in r.get("criterios", []):
                by_id.setdefault(c["id"], []).append(int(c.get("score", 0)))
                max_by_id[c["id"]] = int(c.get("max", 0))
        criterios = [{"id": cid, "score": int(round(sum(v)/len(v))), "max": max_by_id.get(cid, 0), "feedback": []}
                     for cid, v in by_id.items()]
        total = int(round(sum(c["score"] for c in criterios)))
        return {"criterios": criterios, "total": total}

    if stype == "D":
        data = one_call(AIO_PROMPT.format(problem=problem_text, rubric=rubrica_text, snippets=snippets))
        return data

    # fallback -> A
    data = one_call(ONE_STEP_PROMPT.format(problem=problem_text, rubric=rubrica_text, snippets=snippets))
    return data

# ---------- Main ----------
def main():
    # Inputs por env o defaults (pensado para Actions repository_dispatch/workflow_dispatch)
    payload_json = os.getenv("CLIENT_PAYLOAD_JSON")  # opcional: toda la config inline
    if payload_json:
        payload = json.loads(payload_json)
        rubrics_chain = payload.get("rubrics_chain", [])
        rubrics_ref   = payload.get("rubrics_ref", "main")
        scoring       = payload.get("scoring", {"type":"A","ensemble_size":1,"max_context_bytes":120_000,"model":"gpt-5-nano"})
        problem_path  = payload.get("problem_path") or ""
        solution_path = payload.get("solution_path") or ""
        slug          = payload.get("slug","proyecto")
    else:
        # Variables compatibles con el workflow de ejemplo
        rubrics_chain = json.loads(os.getenv("RUBRICS_CHAIN_JSON","[]"))
        rubrics_ref   = os.getenv("RUBRICS_REF","main")
        slug          = os.getenv("PROJECT_SLUG","proyecto")
        scoring = {
            "type": os.getenv("SCORING_TYPE","A"),
            "ensemble_size": int(os.getenv("ENSEMBLE_SIZE","1")),
            "max_context_bytes": int(os.getenv("MAX_CONTEXT_BYTES","250000")),
            "model": os.getenv("MODEL_NAME","gpt-5-nano"),
        }
        problem_path  = os.getenv("PROBLEM_PATH","")
        solution_path = os.getenv("SOLUTION_PATH","")

    # 1) Checkout rubricas ya ocurrió en el workflow -> solo usar ref/info
    log(f"[info] slug={slug} ref={rubrics_ref} stype={scoring['type']} model={scoring.get('model')}")

    # 2) Resolver y fusionar cadena
    yaml = _ensure_yaml()
    merged: Dict[str, Any] = {}
    if not rubrics_chain:
        log("[warn] rubrics_chain vacío; esperando rubrica_efectiva.yaml ya proporcionado.")
        if Path("rubrica_efectiva.yaml").exists():
            merged = yaml.safe_load(Path("rubrica_efectiva.yaml").read_text(encoding="utf-8"))
        else:
            print("ERROR: no hay 'rubrics_chain' ni 'rubrica_efectiva.yaml'", file=sys.stderr)
            sys.exit(2)
    else:
        for raw in rubrics_chain:
            base, spec = parse_ref(raw)
            full = pick_version(base, spec)
            if not full or not Path(full).exists():
                log(f"[warn] Rúbrica no encontrada: {raw}")
                continue
            data = yaml.safe_load(Path(full).read_text(encoding="utf-8"))
            merged = deep_merge(data, merged)
        Path("rubrica_efectiva.yaml").write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
        log("[ok] rubrica_efectiva.yaml generado")

    # Enforce max_context_bytes desde scoring si se pasa
    merged.setdefault("estructura", {}).setdefault("max_context_bytes", int(scoring.get("max_context_bytes", 120_000)))

    # 3) Scanner -> submissions/ + meta + snippets
    scan = run_scanner(merged)
    snippets = scan["snippets"]

    # 4) Cargar enunciado/solución (opcionales). Si no, dummy.
    problem_text = read_text(Path(problem_path)) if problem_path else "Proyecto educativo 4Geeks."
    solution_text = read_text(Path(solution_path)) if solution_path else ""

    # 5) Preparar rubric text (resumen criterios)
    criterios = merged.get("criterios", [])
    rubric_text = json.dumps([{"id":c["id"], "peso":c.get("peso",0), "tipo":c.get("tipo","")} for c in criterios], ensure_ascii=False)

    # 6) Llamar IA según scoring type
    data = evaluate(scoring, rubric_text, problem_text, snippets)

    # 7) Guardar resultados (CSV + Issue)
    criterios_out = data.get("criterios", [])
    total = int(data.get("total", sum(int(c.get("score",0)) for c in criterios_out)))
    peso_total = int(merged.get("peso_total", sum(int(c.get("max", c.get("peso",0))) for c in criterios_out)))
    umbral = int(merged.get("umbrales", {}).get("aprobar", int(0.6 * peso_total)))

    # results.csv (una fila)
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["student_id","logical_marks","syntax_marks","total_marks"])
        w.writeheader()
        # no separamos lógico/sintaxis -> todo en total_marks (ajusta si agregas chequeos deterministas extras)
        w.writerow({"student_id":"student","logical_marks": total, "syntax_marks": 0, "total_marks": total})

    # ISSUE_BODY.md
    lines = [
        f"## Autocorrección — {slug}",
        f"**Branch:** main",
        f"**Score:** {total}/{peso_total} — **Estado:** {'APROBADO ✅' if total>=umbral else 'REVISAR 🔁'} (umbral: {umbral})",
        "",
        "### Desglose",
    ]
    for c in criterios_out:
        lines.append(f"- {c.get('id')}: {c.get('score','?')}/{c.get('max', c.get('peso','?'))}")
    lines += [
        "",
        "🔎 Artefactos adjuntos:",
        "- `results.csv`",
        "- `evaluation.log`",
        "- `scanner_meta.json`",
        "- `rubrica_efectiva.yaml`",
    ]
    Path("ISSUE_BODY.md").write_text("\n".join(lines), encoding="utf-8")
    log("[ok] results.csv e ISSUE_BODY.md listos")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        log(f"[fatal] {e}")
        sys.exit(1)
