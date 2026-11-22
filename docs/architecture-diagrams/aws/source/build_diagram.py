"""Builds the F1 Race Intelligence AWS architecture diagram (draw.io + SVG).

One model, two emitters, so the .drawio and the .svg cannot drift apart.
Run:  python build_diagram.py <output-dir>
Every fact drawn here comes from infra/template.yaml, infra/statemachine/*.asl.json,
infra/aws_status.json, infra/README.md and README.md (see ../README.md for the table).
"""
from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

from PIL import ImageFont

HERE = Path(__file__).resolve().parent
ICON_DIR = HERE / "icons"
W, H = 1920, 1080
FONT = "Segoe UI, Helvetica Neue, Helvetica, Arial, sans-serif"

# ---------------------------------------------------------------- palette (dark)
BG = "#0a101c"
PANEL = "#0e1727"
PANEL_LINE = "#1f2c47"
CARD = "#14223b"
TXT = "#e8eef8"
MUTED = "#9aaac4"
BLUE = "#4da3ff"      # A  defined as IaC
GREEN = "#3ddc97"     # B  validated locally
AMBER = "#ffb020"     # C  real deployment not verified
GREY = "#7d8aa0"      # D  future / pending
PURPLE = "#9b8cff"    # pipeline stages M2-M6
TEAL = "#2fc4a8"      # data lake zones
SALMON = "#e8836a"    # OpenF1 (external)
RED = "#ff7b7b"
C_CTRL, C_DATA, C_ALERT, C_EXT, C_LOG = "#7cb7ff", TEAL, RED, "#ff9b6b", "#8a96a8"

_fonts = {}


def text_width(text: str, size: float, bold: bool) -> float:
    key = (size, bold)
    if key not in _fonts:
        path = "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"
        _fonts[key] = ImageFont.truetype(path, int(round(size * 10)))
    return _fonts[key].getlength(text) / 10.0


WARNINGS: list[str] = []


def L(text, size=11, bold=False, color=TXT, italic=False):
    return dict(text=text, size=size, bold=bold, color=color, italic=italic)


# ---------------------------------------------------------------- model
nodes: list[dict] = []
edges: list[dict] = []
_ids = {"n": 1}


def nid(prefix="c"):
    _ids["n"] += 1
    return f"{prefix}{_ids['n']}"


def rect(x, y, w, h, *, fill=CARD, stroke=BLUE, sw=1.4, dash=None, rx=8, lines=None,
         align="left", pl=10, pr=8, valign="middle", id=None, shape="rect", rot=0,
         alpha=1.0, fit=True):
    n = dict(kind="rect", id=id or nid(), x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, sw=sw,
             dash=dash, rx=rx, lines=lines or [], align=align, pl=pl, pr=pr, valign=valign,
             shape=shape, rot=rot, alpha=alpha)
    if fit and lines:
        avail = (h if rot else w) - pl - pr if not rot else w - pl - pr
        for ln in n["lines"]:
            while text_width(ln["text"], ln["size"], ln["bold"]) > avail * 0.96 and ln["size"] > 9:
                ln["size"] = round(ln["size"] - 0.5, 1)
            if text_width(ln["text"], ln["size"], ln["bold"]) > avail * 0.96:
                WARNINGS.append(f"overflow: {ln['text']!r} in {n['id']} ({avail:.0f}px)")
    nodes.append(n)
    return n


def label(x, y, w, h, lines, align="left", id=None, rot=0):
    return rect(x, y, w, h, fill=None, stroke=None, sw=0, lines=lines, align=align, pl=0, pr=0,
                id=id, rot=rot, rx=0)


def icon(x, y, size, name, id=None):
    n = dict(kind="image", id=id or nid("i"), x=x, y=y, w=size, h=size, icon=name)
    nodes.append(n)
    return n


def badge(cx, cy, r=8):
    rect(cx - r, cy - r, 2 * r, 2 * r, fill=AMBER, stroke="#0a101c", sw=1.2, shape="ellipse",
         lines=[L("?", 11, True, "#1a1200")], align="center", pl=0, pr=0, rx=r, fit=False)


def card(x, y, w, h, icon_name, lines, *, stroke=BLUE, dash=None, id=None, unverified=True,
         icon_size=44, fill=CARD):
    c = rect(x, y, w, h, fill=fill, stroke=stroke, dash=dash, lines=lines, pl=icon_size + 18,
             pr=8, id=id)
    icon(x + 8, y + (h - icon_size) / 2, icon_size, icon_name)
    if unverified:
        badge(x + w - 6, y + 6)
    return c


def chip(x, y, w, h, title, sub, *, stroke, fill=None, dash=None, id=None, tsize=11.5, ssize=10):
    lines = [L(title, tsize, True)]
    if sub:
        lines.append(L(sub, ssize, False, MUTED))
    return rect(x, y, w, h, fill=fill or CARD, stroke=stroke, dash=dash, lines=lines, rx=6,
                pl=9, pr=6, id=id)


def edge(src, tgt, s=(1, .5), t=(0, .5), wps=(), color=C_CTRL, dash=None, sw=1.6, id=None):
    edges.append(dict(id=id or nid("e"), src=src, tgt=tgt, s=s, t=t, wps=list(wps), color=color,
                      dash=dash, sw=sw))


def by_id(i):
    return next(n for n in nodes if n["id"] == i)


def abs_pt(n, f):
    return (n["x"] + f[0] * n["w"], n["y"] + f[1] * n["h"])


# ---------------------------------------------------------------- layout constants
PT, PB = 112, 804                  # column panels
C1, C2, C3, C4, C5 = (30, 260), (325, 400), (770, 380), (1180, 320), (1525, 365)
Y0, PITCH, CH = 500, 38, 32       # aligned rows: task chips <-> zone chips

# ---------------------------------------------------------------- background + header
rect(0, 0, W, H, fill=BG, stroke=None, sw=0, rx=0, id="bg")
label(30, 8, 1180, 32, [L("F1 Race Intelligence · Arquitectura AWS (implementación vigente)", 24, True)], id="title")
label(30, 42, 1190, 16, [L("Fuente de verdad: infra/template.yaml · infra/statemachine/*.asl.json · HEAD efd0a9c "
                           "(docs/record-meeting-key-deployment) · us-east-1 (samconfig dev)", 11, False, MUTED)], id="sub1")
label(30, 59, 1190, 16, [L("Estado: infra/aws_status.json → despliegue DESPLEGADO PARCIALMENTE (reportado) · pruebas en AWS real NO VERIFICABLES · "
                           "PipelineEnabled=false · calendarios DISABLED", 11, True, AMBER)], id="sub2")
label(30, 74, 1190, 14, [L("Runtime 800931b (task def. rev. 2) reportado el 2026-11-22 · smoke 2026-11-21 = rev. 1 / imagen 3d3b4e6 · M2–M6 sin ejecutar con 800931b · "
                           "«?» = reportado, no reproducible desde el repositorio.", 10, False, MUTED)], id="sub3")

# legend
lx, ly = 1230, 10
rect(lx, ly + 2, 24, 14, fill=CARD, stroke=BLUE, rx=3, id="lg_a")
label(lx + 32, ly, 300, 18, [L("A · Definido en IaC (SAM/CloudFormation)", 11)], id="lg_a_t")
badge(lx + 12, ly + 33, 8)
label(lx + 32, ly + 24, 300, 18, [L("C · Reportado en dev (no reproducible aquí)", 11)], id="lg_c_t")
rect(lx + 335, ly + 2, 24, 14, fill=CARD, stroke=GREEN, rx=3, id="lg_b")
label(lx + 367, ly, 300, 18, [L("B · Validado en local / simulación (sin AWS)", 11)], id="lg_b_t")
rect(lx + 335, ly + 26, 24, 14, fill=None, stroke=GREY, dash="1.5 3", rx=3, id="lg_d")
label(lx + 367, ly + 24, 300, 18, [L("D · Futuro / pendiente (no implementado)", 11)], id="lg_d_t")
# edge-type legend
ex = lx
for i, (col, dash, txt) in enumerate([(C_CTRL, None, "control"), (C_DATA, None, "datos S3"),
                                      (C_EXT, None, "OpenF1 HTTPS"), (AMBER, "5 3", "compuerta cerrada"),
                                      (C_ALERT, None, "alerta"), (C_LOG, "2 3", "logs")]):
    xx = ex + i * 112
    rect(xx, ly + 54, 22, 2, fill=col, stroke=None, sw=0, rx=0, id=f"lgl{i}")
    nodes[-1]["_dash_line"] = (col, dash)
    label(xx + 27, ly + 48, 84, 14, [L(txt, 10, False, MUTED)], id=f"lglt{i}")

# ---------------------------------------------------------------- column panels + headers
for (x, w), head, col in [(C1, "1 · INGESTA Y DISPARO", MUTED), (C2, "2 · ORQUESTACIÓN", MUTED),
                           (C3, "3 · PROCESAMIENTO M2–M6", MUTED), (C4, "4 · ALMACENAMIENTO · DATA LAKE", MUTED),
                           (C5, "5 · MODELADO Y SERVING", MUTED)]:
    rect(x, PT, w, PB - PT, fill=PANEL, stroke=PANEL_LINE, sw=1, rx=10, id=f"pan{x}")
    label(x, 90, w, 18, [L(head, 12, True, "#c9d6ec")], id=f"hd{x}")

# ================================================================ COLUMN 1 — ingesta
of1 = rect(42, 120, 236, 46, fill="#241a1a", stroke=SALMON, sw=1.4, rx=8, id="openf1",
           lines=[L("OpenF1 API", 12.5, True), L("servicio externo (no AWS) · api.openf1.org/v1", 10, False, MUTED)])
avail = card(42, 190, 236, 68, "AWSLambda",
             [L("Lambda availability", 11.5, True), L("Python 3.12 · arm64 · 128 MB", 10, False, MUTED),
              L("sin VPC · sin reintentos asíncronos", 10, False, MUTED)], id="avail")
sched = card(42, 430, 236, 68, "AmazonEventBridge",
             [L("EventBridge · regla programada", 11, True), L("rate(15 minutes) por defecto", 10, False, MUTED),
              L("State=DISABLED por defecto", 10, True, AMBER)], id="sched")
label(112, 268, 166, 70, [
    L("Devuelve {available, reason}.", 10, False, MUTED), L("OpenF1 no disponible es una", 10, False, MUTED),
    L("respuesta normal, no un error.", 10, False, MUTED), L("OnFailure → SNS · métricas EMF", 10, False, MUTED),
    L("OpenF1Available / LatencyMs", 10, False, MUTED)], id="avail_note")

edge("avail", "openf1", s=(0.25, 0), t=(0.25, 1), color=C_EXT)
edge("sched", "avail", s=(0.25, 0), t=(0.25, 1), color=C_CTRL)
label(108, 380, 150, 14, [L("invoke asíncrono", 10, False, MUTED)], id="l_inv")
label(108, 168, 170, 14, [L("GET /v1/sessions?session_key=7953", 9.5, False, "#ffb59a")], id="l_get")

# ================================================================ COLUMN 2 — orquestación
bus = card(337, 190, 376, 68, "AmazonEventBridge",
           [L("EventBridge · bus de orquestación", 11.5, True),
            L("regla availability-result → InputPath $.detail.responsePayload", 10, False, MUTED),
            L("reintentos: 3 · edad máx. 900 s", 10, False, MUTED)], id="bus")
adm = card(337, 290, 376, 68, "AWSLambda",
           [L("Lambda pipeline-admission", 11.5, True),
            L("cadencia persistida en S3 (If-None-Match / If-Match)", 10, False, MUTED),
            L("máx. 1 inicio automático por PipelineIntervalHours (24 h)", 10, False, MUTED)], id="adm")
edge("avail", "bus", s=(1, .5), t=(0, .5), color=C_CTRL)
label(284, 207, 52, 14, [L("OnSuccess", 9.5, False, MUTED)], id="l_ons")
edge("bus", "adm", s=(0.5, 1), t=(0.5, 0), color=C_CTRL)
label(530, 262, 170, 14, [L("invoke (regla availability-result)", 9.5, False, MUTED)], id="l_inv2")

sfn_g = rect(337, 372, 376, 426, fill="#0f1a2e", stroke=BLUE, sw=1.4, rx=8, id="sfn")
icon(347, 378, 36, "AWSStepFunctions")
label(391, 376, 300, 44, [L("Step Functions STANDARD · …-dev-pipeline", 11.5, True),
                          L("timeout global 12 h · logs ALL → CloudWatch", 10, False, MUTED),
                          L("referencia la task definition rev. 2", 10, False, MUTED)], id="sfn_t")
badge(713 - 6, 372 + 6)
edge("adm", "sfn", s=(0.5, 1), t=(0.5, 0), color=C_CTRL)
label(530, 361, 170, 12, [L("states:StartExecution", 9.5, False, MUTED)], id="l_start")

rect(347, 422, 356, 22, fill=CARD, stroke=PANEL_LINE, rx=5, pl=9, id="g1",
     lines=[L("IsOpenF1Available · false ⇒ Succeed (no ejecuta nada)", 10.5, False)])
rect(347, 447, 356, 22, fill="#2a2108", stroke=AMBER, rx=5, pl=9, id="g2",
     lines=[L("IsPipelineEnabled · false ⇒ PipelineDisabled (POR DEFECTO)", 10.5, True, AMBER)])
rect(347, 472, 356, 22, fill=CARD, stroke=PANEL_LINE, rx=5, pl=9, id="g3",
     lines=[L("CheckForConcurrentRun → ResolveExtractionScope", 10.5, False)])

s_plan = chip(347, Y0, 356, CH, "PlanSessions + LoadPlan", "runTask.sync plan · LoadPlan: s3:getObject manifests/plan/*",
              stroke=PURPLE, id="s_plan")
s_map = rect(347, Y0 + PITCH, 356, 3 * PITCH + CH, fill=CARD, stroke=PURPLE, sw=1.4, rx=6, id="s_map",
             lines=[L("PerSessionPipeline · Map (INLINE)", 11.5, True),
                    L("una iteración por sesión del plan · MaxConcurrency=1", 10, False, MUTED),
                    L("M2 → M3 → M4 → M5, una tarea Fargate por etapa", 10, False, MUTED),
                    L("cada tarea: Retry + Choice sobre Containers[0].ExitCode", 10, False, MUTED),
                    L("M3 en FAIL se registra, no es fallo de tarea", 10, False, MUTED)], pl=9, pr=6)
s_m6 = chip(347, Y0 + 5 * PITCH, 356, CH, "M6_ModelingDataset", "--require-complete-history · exit 3 ⇒ diferido (Succeed)",
            stroke=PURPLE, id="s_m6")
s_nt = chip(347, Y0 + 6 * PITCH + 2, 356, 34, "NotifyPipelineFailure → SNS", "Catch en cada tarea → PipelineFailed (Fail)",
            stroke=RED, id="s_nt")
label(347, Y0 + 6 * PITCH + 38, 356, 24, [L("Resultado Succeed: OpenF1Unavailable · PipelineDisabled ·", 9.5, False, MUTED),
                                          L("AnotherRunInProgress · PipelineSucceeded · ModelingDeferred", 9.5, False, MUTED)], id="sfn_res")

# ================================================================ COLUMN 3 — procesamiento
ecr = card(782, 170, 300, 66, "AmazonElasticContainerRegistry",
           [L("ECR …-dev-pipeline", 11.5, True), L("imagen 800931b · por digest (reportado)", 10, False, MUTED),
            L("tags inmutables · scan on push", 10, False, MUTED)], id="ecr")
vpc = rect(782, 350, 356, 446, fill="#0f1a2e", stroke=BLUE, sw=1.4, rx=8, id="vpc")
icon(812, 358, 30, "AmazonVirtualPrivateCloud")
label(850, 356, 235, 34, [L("VPC 10.42.0.0/24 · 2 subredes públicas (2 AZ)", 11, True),
                          L("SG: sin ingress · egress solo TCP 443 · sin NAT", 10, False, MUTED)], id="vpc_t")
badge(1138 - 6, 350 + 6)
igw = icon(1086, 330, 28, "AmazonVPCInternetGateway", id="igw")
label(1000, 330, 82, 24, [L("Internet", 9.5, False, MUTED), L("Gateway", 9.5, False, MUTED)], align="right", id="igw_l")

clu = rect(794, 396, 332, 394, fill="#101c33", stroke=BLUE, sw=1.2, rx=8, id="clu")
icon(802, 402, 38, "AWSFargate")
label(846, 400, 240, 58, [L("ECS cluster …-dev-pipeline · Fargate", 11, True),
                          L("task def. rev. 2 · imagen 800931b", 10, False, MUTED),
                          L("512 CPU / 1 GB · x86_64 · sin volúmenes", 10, False, MUTED),
                          L("usuario 10001 · rootfs RO · cap_drop ALL", 10, False, MUTED)], id="clu_t")
badge(1126 - 6, 396 + 6)
gate = rect(806, 464, 276, 22, fill="#2a2108", stroke=AMBER, dash="5 3", sw=1.2, rx=5, id="gate",
            lines=[L("M2–M6 inactivas mientras PipelineEnabled=false", 10, True, AMBER)], align="center", pl=0, pr=0)
CHX, CHW = 820, 294
t_defs = [("plan", "plan · planificación", "sesiones pendientes → manifests/plan/"),
          ("m2", "M2 · extracción", "OpenF1 → raw/ (por sesión, idempotente)"),
          ("m3", "M3 · validación", "valida raw/ → informe PASS/WARNING/FAIL"),
          ("m4", "M4 · consolidación", "raw/ + validated/ → piloto × vuelta"),
          ("m5", "M5 · features", "consolidated/ → features + target"),
          ("m6", "M6 · dataset de modelado", "features/ de todo el histórico → splits")]
for i, (k, t, s) in enumerate(t_defs):
    chip(CHX, Y0 + i * PITCH, CHW, CH, t, s, stroke=PURPLE, id=f"t_{k}")
mapbr = rect(800, Y0 + PITCH - 3, 322, 3 * PITCH + CH + 6, fill=None, stroke=PURPLE, dash="5 3", sw=1.2, rx=8, id="mapbr")
label(739, 604, 140, 14, [L("POR SESIÓN (Map)", 9, True, PURPLE)], align="center", id="maplbl", rot=270)
nodes[-1]["rot_center"] = (809, Y0 + PITCH - 3 + (3 * PITCH + CH + 6) / 2)

# ================================================================ COLUMN 4 — S3
s3g = rect(1190, 120, 300, 686, fill="#0f1a2e", stroke=BLUE, sw=1.4, rx=8, id="s3")
icon(1200, 130, 48, "AmazonSimpleStorageService")
label(1256, 128, 226, 70, [L("S3 data lake", 12.5, True), L("<proyecto>-dev-<cuenta>-<región>", 10, False, MUTED),
                           L("versionado · SSE-S3 (AES256)", 10, False, MUTED),
                           L("acceso público bloqueado", 10, False, MUTED), L("TLS obligatorio · Retain al borrar stack", 10, False, MUTED)], id="s3_t")
badge(1490 - 6, 120 + 6)
label(1202, 206, 276, 64, [L("Lifecycle: raw/ → Intelligent-Tiering a 30 d ·", 10, False, MUTED),
                           L("manifests/ a 90 d · logs/ expira a 365 d ·", 10, False, MUTED),
                           L("versiones no actuales 90 d · nunca expira", 10, False, MUTED),
                           L("los datasets vigentes", 10, False, MUTED)], id="s3_lc")
ctl = chip(1202, 307, 276, 34, "manifests/control/pipeline-cadence.json", "CAS de la Lambda de admisión (única escritora)",
           stroke=TEAL, id="ctl", tsize=10.5, ssize=9.5)
label(1202, 350, 276, 100, [L("Acceso de la tarea de datos (IAM):", 10, True, "#c9d6ec"),
                            L("R/W en raw/ validated/ consolidated/ features/", 10, False, MUTED),
                            L("modeling/ y manifests/{extraction, incremental,", 10, False, MUTED),
                            L("consolidation, features, modeling, plan}", 10, False, MUTED),
                            L("Sin Delete*, sin models/, sin manifests/control/", 10, False, MUTED),
                            L("Salida a S3 por Internet Gateway (sin endpoints)", 10, False, MUTED)], id="s3_iam")
label(1202, 476, 276, 16, [L("ZONAS · layout «lake» (storage/layout.py)", 10.5, True, TEAL)], id="zones_h")
z_defs = [("manifests", "manifests/", "planes y reportes por etapa (6 prefijos)", TEAL, None),
          ("raw", "raw/", "respuestas OpenF1 · inmutable para M3–M6", TEAL, None),
          ("validated", "validated/", "informes de calidad (M3)", TEAL, None),
          ("consolidated", "consolidated/", "piloto × vuelta por sesión (M4)", TEAL, None),
          ("features", "features/", "features + target next_lap_time (M5)", TEAL, None),
          ("modeling", "modeling/", "splits + preprocesador, historia completa (M6)", TEAL, None),
          ("models", "models/", "models/m7 · versiones + latest.json (race model)", PURPLE, "5 3"),
          ("forecasts", "forecasts/", "forecasts/m7 · pronóstico por vuelta (Lambda)", PURPLE, "5 3")]
for i, (k, t, s, col, d) in enumerate(z_defs):
    chip(1202, Y0 + i * PITCH, 276, CH, t, s, stroke=col, dash=d, id=f"z_{k}", tsize=11.5, ssize=9.5)

# ================================================================ COLUMN 5
# B — local validation
rect(1537, 122, 341, 168, fill=PANEL, stroke=GREEN, sw=1.4, rx=8, id="loc")
label(1547, 126, 322, 32, [L("Validado en local · sin AWS", 12, True, GREEN),
                           L("M1–M8 con datos sintéticos y OpenF1 simulado", 10, False, MUTED)], id="loc_t")
for i, (t, s) in enumerate([("M7 entrenamiento + MLflow (SQLite local)", "regresión lineal · registro de modelos"),
                             ("Publicación inmutable del paquete", "published-models/ · manifiesto + PUBLISHED"),
                             ("M8 serving FastAPI (prototipo local)", "/live · /health · /model-info · /predict")]):
    chip(1549, 162 + i * 40, 317, 34, t, s, stroke=GREEN, id=f"loc{i}", tsize=11, ssize=9.5)
label(1549, 282 - 4, 317, 12, [L("tests/unit/test_m7_m8_integration.py · DEMO_LOCAL.md", 9, False, MUTED)], id="loc_f")
# D — future
rect(1537, 298, 341, 128, fill=None, stroke=GREY, dash="1.5 3", sw=1.4, rx=8, id="fut")
label(1549, 303, 320, 18, [L("Pendiente / futuro (README §19)", 12, True, "#b9c4d8")], id="fut_t")
label(1549, 324, 320, 100, [L("· Serving M8 desplegado en AWS (no existe)", 10.5, False, MUTED),
                            L("· MLflow remoto en lugar de SQLite local", 10.5, False, MUTED),
                            L("· Entrenamiento con datos reales de OpenF1", 10.5, False, MUTED),
                            L("· Dashboard de negocio y monitoreo de drift", 10.5, False, MUTED),
                            L("· Despliegue automatizado (CD)", 10.5, False, MUTED)], id="fut_b")
# M7 AWS (defined in IaC, master)
rect(1537, 436, 341, 368, fill=PANEL, stroke=BLUE, sw=1.4, rx=8, id="m7")
label(1547, 440, 322, 46, [L("M7 modelo de carrera · definido en IaC (master)", 11.5, True),
                           L("condicional a ImageUri · calendarios DISABLED", 10, False, MUTED),
                           L("recursos M7 ausentes del stack desplegado (reportado)", 10, False, AMBER)], id="m7_t")


def mini(x, y, name, l1, l2, l3, id):
    rect(x, y, 96, 96, fill=CARD, stroke=BLUE, sw=1.3, rx=8, id=id,
         lines=[L("", 17), L("", 17), L(l1, 10, True), L(l2, 9.5, False, MUTED), L(l3, 9.5, False, MUTED)],
         align="center", pl=2, pr=2, valign="top")
    icon(x + 28, y + 8, 40, name)
    badge(x + 96 - 5, y + 5, 7)


mini(1546, 500, "AWSFargate", "Fargate", "race-model", "2 vCPU · 8 GB", "m7_far")
mini(1664, 500, "AWSStepFunctions", "Step Functions", "race-model", "train | update", "m7_sfn")
mini(1782, 500, "AmazonEventBridge", "EventBridge", "train · update", "DISABLED", "m7_eb1")
mini(1664, 640, "AWSLambda", "Lambda", "race-forecast", "imagen · 3008 MB", "m7_lam")
mini(1782, 640, "AmazonEventBridge", "EventBridge", "live rate(1 min)", "DISABLED", "m7_eb2")
edge("m7_eb1", "m7_sfn", s=(0, .5), t=(1, .5), color=C_CTRL)
edge("m7_sfn", "m7_far", s=(0, .5), t=(1, .5), color=C_CTRL)
edge("m7_eb2", "m7_lam", s=(0, .5), t=(1, .5), color=C_CTRL)

# ================================================================ edges: processing/S3/OpenF1
edge("s_plan", "t_plan", s=(1, .5), t=(0, .5), color=AMBER, dash="5 3")
edge("s_map", "mapbr", s=(1, .5), t=(0, .5), color=AMBER, dash="5 3")
edge("s_m6", "t_m6", s=(1, .5), t=(0, .5), color=AMBER, dash="5 3")
label(716, Y0 - 1, 64, 14, [L("runTask.sync", 9, False, AMBER)], id="l_rts", align="center")
for t, z in [("plan", "manifests"), ("m2", "raw"), ("m3", "validated"), ("m4", "consolidated"),
             ("m5", "features"), ("m6", "modeling")]:
    edge(f"t_{t}", f"z_{z}", s=(1, .5), t=(0, .5), color=C_DATA)
edge("t_plan", "igw", s=(280/294, 0), t=(0.5, 1), color=C_EXT)
# corridor to OpenF1: IGW top -> up -> left -> OpenF1 right edge
edge("igw", "openf1", s=(0.5, 0), t=(1, .5), wps=[(1100, 143)], color=C_EXT)
label(420, 125, 640, 14, [L("HTTPS 443 · plan (catálogo) y M2 → OpenF1, vía Internet Gateway (sin NAT ni endpoints)", 10, False, "#ffb59a")],
      align="center", id="l_corr")
edge("ecr", "clu", s=(18/300, 1), t=(6/332, 0), color=C_CTRL)
label(808, 292, 110, 12, [L("pull por digest", 9.5, False, MUTED)], id="l_pull")
edge("adm", "ctl", s=(1, .5), t=(0, .5), color=C_DATA)
label(722, 308, 120, 14, [L("CAS cadence.json", 9.5, False, MUTED)], id="l_cas")
edge("m7_far", "z_models", s=(0.5, 1), t=(1, .5), wps=[(1594, 744)], color=C_DATA)
label(1600, 650, 60, 26, [L("escribe", 9, False, MUTED), L("models/m7", 9, False, MUTED)], id="l_mw")
edge("m7_lam", "z_forecasts", s=(0.5, 1), t=(1, .5), wps=[(1712, 782)], color=C_DATA)
label(1722, 744, 150, 26, [L("lee raw/ y models/m7", 9, False, MUTED), L("escribe forecasts/m7", 9, False, MUTED)], id="l_fw")

# ================================================================ bottom bands
OB, OBH = 826, 126
rect(30, OB, 1860, OBH, fill=PANEL, stroke=PANEL_LINE, sw=1, rx=10, id="obs")
label(42, OB + 2, 400, 16, [L("6 · OBSERVABILIDAD Y ALERTAS", 12, True, "#c9d6ec")], id="obs_h")
OY = OB + 22
o1 = rect(42, OY, 1090, 98, fill="#0f1a2e", stroke=BLUE, sw=1.3, rx=8, id="o1")
icon(50, OY + 6, 32, "AmazonCloudWatch")
label(90, OY + 4, 700, 18, [L("CloudWatch · Logs (LogRetentionDays=30), métricas y dashboard", 11.5, True)], id="o1_t")
badge(1132 - 6, OY + 6)
logs = [("/aws/lambda/…-openf1-availability", BLUE, None), ("/aws/lambda/…-pipeline-admission", BLUE, None),
        ("/aws/vendedlogs/states/…-pipeline", BLUE, None), ("/aws/ecs/…-pipeline (tareas M2–M6)", BLUE, None),
        ("Dashboard …-operations (8 widgets)", BLUE, None), ("Métricas EMF OpenF1Available / LatencyMs", BLUE, None),
        ("M7 · /aws/ecs/…-race-model", BLUE, "4 3"), ("M7 · logs race-forecast y states race-model", BLUE, "4 3")]
for i, (t, c, d) in enumerate(logs):
    cx, cy = 50 + (i % 4) * 268, OY + 30 + (i // 4) * 32
    rect(cx, cy, 260, 25, fill=CARD, stroke=c, dash=d, sw=1.1, rx=5, lines=[L(t, 10.5, False)], pl=8, pr=4, id=f"lg{i}")
o2 = rect(1146, OY, 390, 98, fill="#0f1a2e", stroke=BLUE, sw=1.3, rx=8, id="o2")
icon(1154, OY + 4, 24, "AmazonCloudWatchAlarm")
label(1184, OY + 4, 330, 16, [L("CloudWatch · Alarmas (OK/ALARM → SNS)", 11.5, True)], id="o2_t")
badge(1536 - 6, OY + 6)
alarms = ["availability-checker-errors", "availability-checker-throttles", "pipeline-admission-errors",
          "pipeline-executions-failed", "orchestration-rules-failed", "openf1-unavailable-sustained (6 h)",
          "M7 · race-model-executions-failed", "M7 · race-forecast-errors"]
for i, t in enumerate(alarms):
    cx, cy = 1154 + (i % 2) * 190, OY + 26 + (i // 2) * 17.5
    rect(cx, cy, 184, 16, fill=CARD, stroke=BLUE, dash="4 3" if t.startswith("M7") else None, sw=1, rx=4,
         lines=[L(t, 9.5, False)], pl=6, pr=2, id=f"al{i}")
o3 = card(1558, OY, 332, 98, "AmazonSimpleNotificationService",
          [L("SNS …-dev-critical-alerts", 11.5, True), L("e-mail opcional (AlertEmail; vacío ⇒ sin destinatario)", 9.5, False, MUTED),
           L("publican: alarmas CloudWatch, OnFailure de Lambdas,", 9.5, False, MUTED),
           L("NotifyPipelineFailure (SFN) y fallos de race-model", 9.5, False, MUTED)], id="o3", icon_size=40)
edge("o2", "o3", s=(1, .5), t=(0, .5), color=C_ALERT)
for col, c in [("pan30", 262), ("pan325", 525), ("pan770", 960)]:
    edge(col, "o1", s=((c - by_id(col)["x"]) / by_id(col)["w"], 1), t=((c - 42) / 1090, 0), color=C_LOG, dash="2 3", sw=1.4)
label(270, 809, 100, 14, [L("logs y métricas", 9, False, MUTED)], id="l_lg1")

SB = OB + OBH + 8
rect(30, SB, 1860, 1070 - SB, fill=PANEL, stroke=PANEL_LINE, sw=1, rx=10, id="sec")
label(42, SB + 2, 400, 16, [L("7 · SEGURIDAD", 12, True, "#c9d6ec")], id="sec_h")
SY = SB + 22
rect(42, SY, 960, 1070 - SY - 8, fill="#0f1a2e", stroke=BLUE, sw=1.3, rx=8, id="s1")
icon(50, SY + 6, 30, "AWSIdentityandAccessManagement")
label(88, SY + 4, 880, 18, [L("IAM · mínimo privilegio · un rol por responsabilidad · sin políticas administradas ni AdministratorAccess", 11, True)], id="s1_t")
badge(1002 - 6, SY + 6)
roles1 = ["availability-lambda · sin S3", "pipeline-admission · cadence.json", "pipeline-states · RunTask + PassRole",
          "data-processing · tarea S3 R/W", "data-processing-exec · ECR + logs"]
roles2 = ["race-model · R raw, R/W models", "race-model-exec · ECR + logs", "race-model-states · RunTask",
          "race-model-schedule · Start", "race-forecast · R raw, W forecasts"]
for r, roles in enumerate([roles1, roles2]):
    for i, t in enumerate(roles):
        rect(50 + i * 189, SY + 28 + r * 26, 184, 21, fill=CARD, stroke=BLUE, dash="4 3" if r else None, sw=1.1, rx=5,
             lines=[L(t, 9.5, False)], pl=6, pr=2, id=f"ro{r}{i}")
rect(1014, SY, 440, 1070 - SY - 8, fill="#0f1a2e", stroke=BLUE, sw=1.3, rx=8, id="s2",
     lines=[L("Datos y red", 11, True), L("S3: SSE-S3 · versionado · acceso público bloqueado · TLS", 10, False, MUTED),
            L("Red: VPC dedicada · SG sin ingress · egress solo TCP 443", 10, False, MUTED),
            L("ECR: tags inmutables · scan on push · imagen por digest", 10, False, MUTED),
            L("Confianza con aws:SourceAccount (SFN, ECS, EventBridge)", 10, False, MUTED)], pl=12, valign="middle")
badge(1454 - 6, SY + 6)
rect(1466, SY, 424, 1070 - SY - 8, fill="#2a2108", stroke=AMBER, sw=1.3, rx=8, id="s3b",
     lines=[L("Valores seguros por defecto (contrato)", 11, True, AMBER),
            L("PipelineEnabled=false · AvailabilityScheduleState=DISABLED", 10, False, "#f2d9a0"),
            L("ModelScheduleState / LiveForecastScheduleState = DISABLED", 10, False, "#f2d9a0"),
            L("confirm_changeset=true · imágenes por digest, no por tag", 10, False, "#f2d9a0"),
            L("Regla: PipelineEnabled=true exige PipelineImageUri", 10, False, "#f2d9a0")], pl=12, valign="middle")


# ---------------------------------------------------------------- emitters
def icon_b64(name):
    return base64.b64encode((ICON_DIR / f"{name}.svg").read_bytes()).decode()


def path_points(e):
    s = abs_pt(by_id(e["src"]), e["s"])
    t = abs_pt(by_id(e["tgt"]), e["t"])
    return [s] + e["wps"] + [t]


def line_box(n):
    lines = n["lines"]
    return sum(l["size"] * 1.32 for l in lines)


def svg_text(n):
    out = []
    lines = n["lines"]
    if not lines:
        return ""
    total = line_box(n)
    if n["valign"] == "middle":
        top = n["y"] + (n["h"] - total) / 2
    else:
        top = n["y"] + 5
    if n["align"] == "center":
        x, anchor = n["x"] + n["w"] / 2 + (n["pl"] - n["pr"]) / 2, "middle"
    else:
        x, anchor = n["x"] + n["pl"], "start"
    spans = []
    for ln in lines:
        lh = ln["size"] * 1.32
        base = top + lh / 2 + ln["size"] * 0.35
        top += lh
        if not ln["text"]:
            continue
        spans.append(
            f'<text x="{x:.1f}" y="{base:.1f}" text-anchor="{anchor}" font-size="{ln["size"]}" '
            f'font-weight="{"700" if ln["bold"] else "400"}" fill="{ln["color"]}">{html.escape(ln["text"])}</text>')
    g = "".join(spans)
    if n["rot"]:
        cx, cy = n.get("rot_center", (n["x"] + n["w"] / 2, n["y"] + n["h"] / 2))
        # text boxes with rotation are drawn about their own centre
        return f'<g transform="rotate({-90 if n["rot"] == 270 else n["rot"]} {cx:.1f} {cy:.1f})">{g}</g>'
    return g


def to_svg():
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
         '<title>F1 Race Intelligence - Arquitectura AWS</title>',
         '<defs>' + "".join(
             f'<marker id="ar{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
             f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
             for i, c in enumerate([C_CTRL, C_DATA, C_EXT, AMBER, C_ALERT, C_LOG])) + '</defs>']
    marker_of = {c: i for i, c in enumerate([C_CTRL, C_DATA, C_EXT, AMBER, C_ALERT, C_LOG])}
    for n in nodes:
        if n["kind"] == "image":
            p.append(f'<image x="{n["x"]:.1f}" y="{n["y"]:.1f}" width="{n["w"]}" height="{n["h"]}" '
                     f'href="data:image/svg+xml;base64,{icon_b64(n["icon"])}"/>')
            continue
        if n["fill"] or n["stroke"]:
            da = f' stroke-dasharray="{n["dash"]}"' if n["dash"] else ""
            if n["shape"] == "ellipse":
                p.append(f'<ellipse cx="{n["x"] + n["w"] / 2}" cy="{n["y"] + n["h"] / 2}" rx="{n["w"] / 2}" ry="{n["h"] / 2}" '
                         f'fill="{n["fill"]}" stroke="{n["stroke"]}" stroke-width="{n["sw"]}"/>')
            else:
                fill = n["fill"] or "none"
                stroke = n["stroke"] or "none"
                p.append(f'<rect x="{n["x"]}" y="{n["y"]}" width="{n["w"]}" height="{n["h"]}" rx="{n["rx"]}" '
                         f'fill="{fill}" stroke="{stroke}" stroke-width="{n["sw"]}"{da}/>')
            if n.get("_dash_line"):
                col, d = n["_dash_line"]
                if d:
                    p[-1] = (f'<line x1="{n["x"]}" y1="{n["y"] + 1}" x2="{n["x"] + n["w"]}" y2="{n["y"] + 1}" '
                             f'stroke="{col}" stroke-width="2" stroke-dasharray="{d}"/>')
        if (n["fill"] or n["stroke"]) and n["shape"] != "ellipse":
            p.append(svg_text(n))
    # edges drawn after shapes, before nothing else (labels were declared as nodes; re-draw them on top)
    for e in edges:
        pts = path_points(e)
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        da = f' stroke-dasharray="{e["dash"]}"' if e["dash"] else ""
        p.append(f'<path d="{d}" fill="none" stroke="{e["color"]}" stroke-width="{e["sw"]}"{da} '
                 f'marker-end="url(#ar{marker_of[e["color"]]})"/>')
    # labels + badges on top of edges
    for n in nodes:
        if n["kind"] == "rect" and (n["fill"] is None and n["stroke"] is None) and n["lines"]:
            p.append(svg_text(n))
        if n["kind"] == "rect" and n["shape"] == "ellipse":
            p.append(svg_text(n))
    p.append("</svg>")
    return "\n".join(p)


def dio_label(n):
    parts = []
    for ln in n["lines"]:
        t = html.escape(ln["text"]) if ln["text"] else "&nbsp;"
        t = f'<b>{t}</b>' if ln["bold"] else t
        parts.append(f'<font style="font-size:{ln["size"]}px" color="{ln["color"]}">{t}</font>')
    return "<br>".join(parts)


def to_drawio():
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    # draw.io z-order = document order: shapes first, then edges, then free labels/badges
    def vertex(n):
        if n["kind"] == "image":
            st = f'shape=image;imageAspect=0;aspect=fixed;image=data:image/svg+xml,{icon_b64(n["icon"])};'
            val = ""
        else:
            fill = f'fillColor={n["fill"]};' if n["fill"] else "fillColor=none;"
            stroke = f'strokeColor={n["stroke"]};strokeWidth={n["sw"]};' if n["stroke"] else "strokeColor=none;"
            dash = f'dashed=1;dashPattern={n["dash"]};' if n["dash"] else ""
            shp = "ellipse;" if n["shape"] == "ellipse" else (f'rounded=1;arcSize={max(0, min(50, n["rx"] * 100 / min(n["w"], n["h"])))};' if n["rx"] else "")
            al = "center" if n["align"] == "center" else "left"
            va = "middle" if n["valign"] == "middle" else "top"
            rot = f'rotation={n["rot"]};' if n["rot"] else ""
            st = (f'{shp}whiteSpace=nowrap;html=1;{fill}{stroke}{dash}{rot}align={al};verticalAlign={va};'
                  f'spacingLeft={n["pl"]};spacingRight={n["pr"]};spacingTop=3;fontFamily=Segoe UI;fontColor={TXT};')
            val = dio_label(n)
        x, y, w, h = n["x"], n["y"], n["w"], n["h"]
        if n["kind"] == "rect" and n["rot"]:
            cx, cy = n.get("rot_center", (x + w / 2, y + h / 2))
            x, y = cx - w / 2, cy - h / 2
        return (f'<mxCell id="{n["id"]}" value="{html.escape(val, quote=True)}" style="{st}" vertex="1" parent="1">'
                f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    top = [n for n in nodes if n["kind"] == "rect" and ((n["fill"] is None and n["stroke"] is None and n["lines"]) or n["shape"] == "ellipse")]
    base = [n for n in nodes if n not in top]
    for n in base:
        cells.append(vertex(n))
    for e in edges:
        dash = f'dashed=1;dashPattern={e["dash"]};' if e["dash"] else ""
        st = (f'edgeStyle=none;html=1;rounded=0;endArrow=block;endFill=1;strokeColor={e["color"]};strokeWidth={e["sw"]};{dash}'
              f'exitX={e["s"][0]:.4f};exitY={e["s"][1]:.4f};exitPerimeter=0;entryX={e["t"][0]:.4f};entryY={e["t"][1]:.4f};entryPerimeter=0;')
        pts = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in e["wps"])
        arr = f'<Array as="points">{pts}</Array>' if pts else ""
        cells.append(f'<mxCell id="{e["id"]}" style="{st}" edge="1" parent="1" source="{e["src"]}" target="{e["tgt"]}">'
                     f'<mxGeometry relative="1" as="geometry">{arr}</mxGeometry></mxCell>')
    for n in top:
        cells.append(vertex(n))
    body = "\n        ".join(cells)
    return ('<mxfile host="app.diagrams.net" agent="build_diagram.py" version="24.0.0">\n'
            '  <diagram id="f1-aws-architecture" name="F1 Race Intelligence - AWS">\n'
            f'    <mxGraphModel dx="{W}" dy="{H}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
            f'page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" background="{BG}" math="0" shadow="0">\n'
            f'      <root>\n        {body}\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n')


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    stem = sys.argv[2] if len(sys.argv) > 2 else "f1-race-intelligence-aws-architecture"
    (out / f"{stem}.svg").write_text(to_svg(), encoding="utf-8")
    (out / f"{stem}.drawio").write_text(to_drawio(), encoding="utf-8")
    print("nodes", len(nodes), "edges", len(edges))
    for w in WARNINGS:
        print("WARN", w)
