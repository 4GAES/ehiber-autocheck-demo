#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto Evaluator - Versión IA (GPT-5 nano)
Genera una rúbrica efectiva combinando los YAML del proyecto mediante IA,
evalúa el repositorio y deja un informe/issue.

Compatible con GitHub Actions.
"""

import os, sys, json, yaml, re, base64, time
from pathlib import Path
from urllib.parse import urlparse
import urllib.request

# =====================================================
# UTILIDADES GENERALES
# =====================================================

def log(msg: str):
    print(msg, flush=True)

def gh_headers():
    token = os.getenv("GITHUB_TOKEN") or os.getenv("KLAUS")
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def gh_get_json(url):
    req = urllib.request.Request(url, headers=gh_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def path_exists(owner, repo, branch, path):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    try:
        req = urllib.request.Request(url, headers=gh_headers())
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status == 200
    except:
        return False

def fetch_text(owner, repo, branch, path, max_bytes=200_000):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    try:
        req = urllib.request.Request(url, headers=gh_headers())
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, dict) and data.get("encoding") == "base64":
            raw = base64.b64decode(data["content"])
            return raw[:max_bytes].decode("utf-8", errors="ignore")
    except Exception:
        pass
    return ""

def parse_repo(repo_url: str):
    u = urlparse(repo_url)
    parts = [p for p in u.path.split("/") if p]
    return parts[0], parts[1].removesuffix(".git")

# =====================================================
# BLOQUE IA PARA FUSIÓN DE YAMLs
# =====================================================


def _openai_chat_llm(model: str, messages, temperature=0.2) -> str:
    import urllib.request, json, re
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("KLAUS")
    if not api_key:
        raise RuntimeError("❌ Falta OPENAI_API_KEY o KLAUS")

    try:
        # Limpieza preventiva
        clean_msgs = []
        for m in messages:
            content = str(m.get("content", "")).encode("utf-8", "ignore").decode("utf-8", "ignore")
            content = re.sub(r"[\x00-\x1f]+", " ", content)
            clean_msgs.append({
                "role": m.get("role", "user"),
                "content": content
            })

        # Construcción del cuerpo
        body = {"model": model, "messages": clean_msgs}
        if not model.startswith("gpt-5-nano"):  # ⚙️ no enviar temperature si es nano
            body["temperature"] = float(temperature)

        print("[debug] model:", model)
        print("[debug] body sample:", json.dumps(body, ensure_ascii=False)[:800], flush=True)

        data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        req = urllib.request.Request("https://api.openai.com/v1/chat/completions")
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, data, timeout=180) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            out = json.loads(raw)
            return out["choices"][0]["message"]["content"]

    except urllib.error.HTTPError as e:
        try:
            err_text = e.read().decode("utf-8", errors="ignore")
            print(f"[fatal] OpenAI API HTTPError {e.code}: {err_text}", flush=True)
        except Exception as inner:
            print(f"[fatal] OpenAI API HTTPError {e.code} (sin cuerpo): {inner}", flush=True)
        raise

    except Exception as e:
        print(f"[fatal] Error general en _openai_chat_llm: {e}", flush=True)
        raise


def generate_effective_rubric_from_yamls(model: str, slug: str, learn_meta: dict, yaml_texts: dict) -> str:
    system = (
        "Eres un generador de rúbricas YAML para autocorrección de proyectos. "
        "Debes devolver SOLO YAML válido (sin bloques ```), con claves top-level: "
        "estructura, criterios, peso_total, umbrales. No devuelvas explicaciones."
    )
    user = f"""
Genera una rúbrica YAML efectiva para el proyecto "{slug}" fusionando estas capas (de más específica a más global):

- module.yaml:
{yaml_texts.get('module','(VACIO)')}

- stack.yaml:
{yaml_texts.get('stack','(VACIO)')}

- global.yaml:
{yaml_texts.get('global','(VACIO)')}

### Reglas de fusión:
1. Precedencia: module > stack > global.
2. Estructura: concatena pick_rules y une required_files / required_any_of sin duplicados.
3. Criterios: merge por id (tipo/peso del más específico).
4. peso_total: suma de pesos si no se define; umbral.aprobar: 60% del peso_total por defecto.
5. Devuelve YAML limpio y válido UTF-8.
6. No incluyas explicaciones ni ```yaml.

### learn.json (contexto):
{json.dumps(learn_meta, ensure_ascii=False, indent=2)}
"""
    content = _openai_chat_llm(model, [
        {"role":"system","content":system},
        {"role":"user","content":user}
    ])
    m = re.search(r"```yaml\s*(.*?)```", content, flags=re.S)
    if m:
        content = m.group(1)
    return content.strip()

# =====================================================
# EVALUACIÓN DEMO (simplificada)
# =====================================================

def ia_stub_eval(criterio):
    base = round(criterio.get("peso",1) * 0.7, 2)
    feedback = [
        "Buen uso de la estructura solicitada.",
        "Revisa pequeños detalles de presentación.",
        "Cumple con los requisitos esenciales."
    ]
    return {
        "id": criterio.get("id","?"),
        "tipo":"ia",
        "score": base,
        "max": criterio.get("peso",1),
        "logs": feedback
    }

def deterministic_stub_eval(criterio, owner, repo, branch):
    peso = criterio.get("peso",1)
    score = peso
    logs = []
    for chk in criterio.get("checks", []):
        if "path_must_exist" in chk:
            for p in chk["path_must_exist"]:
                ok = path_exists(owner, repo, branch, p)
                logs.append(f"exists:{p}={ok}")
                if not ok: score -= peso * 0.5
    return {
        "id": criterio.get("id","?"),
        "tipo":"determinista",
        "score": max(score,0),
        "max": peso,
        "logs": logs
    }

# =====================================================
# MAIN
# =====================================================

def main():
    log("=== 🚀 Auto Evaluator (GPT-5 nano) ===")

    # Leer payload (simulado o real)
    payload_json = os.getenv("CLIENT_PAYLOAD_JSON","{}")
    payload = json.loads(payload_json)

    slug = payload.get("slug","unknown")
    rubrics_chain = payload.get("rubrics_chain",[])
    scoring = payload.get("scoring",{})
    ref = payload.get("rubrics_ref","main")

    log(f"[info] slug={slug} ref={ref} stype={scoring.get('type','A')} model={scoring.get('model','gpt-5-nano')}")

    use_llm = os.getenv("USE_LLM_RUBRIC","0") == "1"
    
    # =====================================================
    # GENERAR RUBRICA EFECTIVA (IA)
    # =====================================================
    if use_llm:
        def _read_or_empty(path):
            p = Path(os.path.join("rubricas", path))
            return p.read_text(encoding="utf-8") if p.exists() else ""

        module_path = next((c for c in rubrics_chain if c.startswith("modules/")), "")
        stack_path  = next((c for c in rubrics_chain if c.startswith("stacks/")), "")
        global_path = next((c for c in rubrics_chain if c.startswith("globals/")), "")

        yaml_texts = {
            "module": _read_or_empty(module_path),
            "stack":  _read_or_empty(stack_path),
            "global": _read_or_empty(global_path),
        }

        learn_meta = {"slug": slug, "scoring": scoring}
        print("[debug] module:", len(yaml_texts["module"]), "bytes")
        print("[debug] stack:", len(yaml_texts["stack"]), "bytes")
        print("[debug] global:", len(yaml_texts["global"]), "bytes")
        rubrica_yaml = generate_effective_rubric_from_yamls(
            scoring.get("model","gpt-5-nano"), slug, learn_meta, yaml_texts
        )

        Path("rubrica_efectiva.yaml").write_text(rubrica_yaml, encoding="utf-8")
        log("[ok] rubrica_efectiva.yaml generado")

        # Validar YAML
        try:
            rub = yaml.safe_load(rubrica_yaml)
            if not isinstance(rub, dict) or "criterios" not in rub:
                raise ValueError("rubrica sin clave 'criterios'")
            log("[ok] rubrica_efectiva.yaml corregido y válido")
        except Exception as e:
            log(f"[fatal] Error leyendo rubrica_efectiva.yaml → {e}")
            sys.exit(1)
    else:
        log("[fatal] Modo IA desactivado y no hay lógica alternativa.")
        sys.exit(1)

    # =====================================================
    # EVALUACIÓN DEMO (IA + determinista)
    # =====================================================
    owner, repo = "demo", "demo"
    branch = "main"

    resultados = []
    total = 0
    for c in rub.get("criterios", []):
        if c.get("tipo") == "determinista":
            r = deterministic_stub_eval(c, owner, repo, branch)
        else:
            r = ia_stub_eval(c)
        total += r["score"]
        resultados.append(r)

    peso_total = rub.get("peso_total", sum(c.get("peso",1) for c in rub.get("criterios",[])))
    umbral = rub.get("umbrales", {}).get("aprobar", round(peso_total * 0.6, 2))
    estado = "APROBADO ✅" if total >= umbral else "REVISAR 🔁"

    # =====================================================
    # SALIDA / INFORME
    # =====================================================
    desglose = "\n".join([f"- {r['id']}: {r['score']}/{r['max']}" for r in resultados])
    feedback = "\n".join([f"- ({r['id']}) {r['logs'][0]}" for r in resultados if r.get("logs")])

    cuerpo = f"""## Resultado módulo {slug}

**Score:** {round(total,2)}/{peso_total}
**Estado:** {estado} (Umbral: {umbral})

### Desglose por criterios
{desglose}

### Feedback breve
{feedback or "- sin observaciones -"}
"""

    # Guardar informe.json (artefacto)
    Path("informe.json").write_text(
        json.dumps({
            "slug": slug,
            "score_total": total,
            "peso_total": peso_total,
            "estado": estado,
            "criterios": resultados
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    log("💾 informe.json guardado")

    # 👉 Crear ISSUE_BODY.md para que el paso con gh lo use
    Path("ISSUE_BODY.md").write_text(cuerpo, encoding="utf-8")
    log("📝 ISSUE_BODY.md generado")

    # (Opcional) publicar issue directo desde Python — dejar comentado
    try:
        issue_url = publicar_issue(owner, repo, f"[AutoEval] {slug}", cuerpo)
        log(f"✅ Issue creado: {issue_url}")
    except Exception as e:
        log(f"[warn] No se pudo crear Issue desde Python: {e}")

    # Imprimir el cuerpo por consola (útil en logs)
    print(cuerpo)


    # (Opcional) publicar issue si tienes permisos:
    try:
        issue_url = publicar_issue(owner, repo, f"[AutoEval] {slug}", cuerpo)
        log(f"✅ Issue creado: {issue_url}")
    except Exception as e:
        log(f"[warn] No se pudo crear Issue: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[fatal] {e}")
        sys.exit(1)
