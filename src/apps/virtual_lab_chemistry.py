# apps/virtual_lab_chemistry.py
import cv2, mediapipe as mp, time
def draw_text(frame, text, pos, size=1.0, color=(255, 255, 255), thickness=2):
    """Draw text with flexible size and color."""
    x, y = pos
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size, color, thickness, cv2.LINE_AA
    )

# ---------- CONFIG ----------
PINCH_THRESHOLD = 40

menus_chemistry = {
    "Chemistry Lab": ["Chemical Reactions", "Periodic Table", "Reaction Balancer", "Back"],
}

# ---------- STATE ----------
chemistry_menu = "Chemistry Lab"
chemistry_selected = -1

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
)
# apps/virtual_lab_chemistry.py
# Full interactive periodic table + chemistry lab mini-features (non-OOP)
# Requires: OpenCV (cv2) and MediaPipe (mediapipe)

import cv2
import mediapipe as mp
import time
import math
import sys

# -------------------- CONFIG --------------------
PINCH_THRESHOLD = 40
SCREEN_W = 1280
SCREEN_H = 720

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5,
                       min_tracking_confidence=0.5,
                       max_num_hands=1)










# -------------------- DATA: ELEMENTS 1..118 --------------------
# Each element: symbol, name, atomic mass (approx), group, period
ELEMENTS = {
  1:  {"symbol":"H",  "name":"Hydrogen",        "mass":1.008,     "group":1,  "period":1},
  2:  {"symbol":"He", "name":"Helium",          "mass":4.002602,  "group":18, "period":1},
  3:  {"symbol":"Li", "name":"Lithium",         "mass":6.94,      "group":1,  "period":2},
  4:  {"symbol":"Be", "name":"Beryllium",       "mass":9.0121831, "group":2,  "period":2},
  5:  {"symbol":"B",  "name":"Boron",           "mass":10.81,     "group":13, "period":2},
  6:  {"symbol":"C",  "name":"Carbon",          "mass":12.011,    "group":14, "period":2},
  7:  {"symbol":"N",  "name":"Nitrogen",        "mass":14.007,    "group":15, "period":2},
  8:  {"symbol":"O",  "name":"Oxygen",          "mass":15.999,    "group":16, "period":2},
  9:  {"symbol":"F",  "name":"Fluorine",        "mass":18.998403, "group":17, "period":2},
 10:  {"symbol":"Ne", "name":"Neon",            "mass":20.1797,   "group":18, "period":2},
 11:  {"symbol":"Na", "name":"Sodium",          "mass":22.989769, "group":1,  "period":3},
 12:  {"symbol":"Mg", "name":"Magnesium",       "mass":24.305,    "group":2,  "period":3},
 13:  {"symbol":"Al", "name":"Aluminium",       "mass":26.981538, "group":13, "period":3},
 14:  {"symbol":"Si", "name":"Silicon",         "mass":28.085,    "group":14, "period":3},
 15:  {"symbol":"P",  "name":"Phosphorus",      "mass":30.973762, "group":15, "period":3},
 16:  {"symbol":"S",  "name":"Sulfur",          "mass":32.06,     "group":16, "period":3},
 17:  {"symbol":"Cl", "name":"Chlorine",        "mass":35.45,     "group":17, "period":3},
 18:  {"symbol":"Ar", "name":"Argon",           "mass":39.948,    "group":18, "period":3},
 19:  {"symbol":"K",  "name":"Potassium",       "mass":39.0983,   "group":1,  "period":4},
 20:  {"symbol":"Ca", "name":"Calcium",         "mass":40.078,    "group":2,  "period":4},
 21:  {"symbol":"Sc", "name":"Scandium",        "mass":44.955908, "group":3,  "period":4},
 22:  {"symbol":"Ti", "name":"Titanium",        "mass":47.867,    "group":4,  "period":4},
 23:  {"symbol":"V",  "name":"Vanadium",        "mass":50.9415,   "group":5,  "period":4},
 24:  {"symbol":"Cr", "name":"Chromium",        "mass":51.9961,   "group":6,  "period":4},
 25:  {"symbol":"Mn", "name":"Manganese",       "mass":54.938044, "group":7,  "period":4},
 26:  {"symbol":"Fe", "name":"Iron",            "mass":55.845,    "group":8,  "period":4},
 27:  {"symbol":"Co", "name":"Cobalt",          "mass":58.933194, "group":9,  "period":4},
 28:  {"symbol":"Ni", "name":"Nickel",          "mass":58.6934,   "group":10, "period":4},
 29:  {"symbol":"Cu", "name":"Copper",          "mass":63.546,    "group":11, "period":4},
 30:  {"symbol":"Zn", "name":"Zinc",            "mass":65.38,     "group":12, "period":4},
 31:  {"symbol":"Ga", "name":"Gallium",         "mass":69.723,    "group":13, "period":4},
 32:  {"symbol":"Ge", "name":"Germanium",       "mass":72.630,    "group":14, "period":4},
 33:  {"symbol":"As", "name":"Arsenic",         "mass":74.921595, "group":15, "period":4},
 34:  {"symbol":"Se", "name":"Selenium",        "mass":78.971,    "group":16, "period":4},
 35:  {"symbol":"Br", "name":"Bromine",         "mass":79.904,    "group":17, "period":4},
 36:  {"symbol":"Kr", "name":"Krypton",         "mass":83.798,    "group":18, "period":4},
 37:  {"symbol":"Rb", "name":"Rubidium",        "mass":85.4678,   "group":1,  "period":5},
 38:  {"symbol":"Sr", "name":"Strontium",       "mass":87.62,     "group":2,  "period":5},
 39:  {"symbol":"Y",  "name":"Yttrium",         "mass":88.90584,  "group":3,  "period":5},
 40:  {"symbol":"Zr", "name":"Zirconium",       "mass":91.224,    "group":4,  "period":5},
 41:  {"symbol":"Nb", "name":"Niobium",         "mass":92.90637,  "group":5,  "period":5},
 42:  {"symbol":"Mo", "name":"Molybdenum",      "mass":95.95,     "group":6,  "period":5},
 43:  {"symbol":"Tc", "name":"Technetium",      "mass":98,        "group":7,  "period":5},
 44:  {"symbol":"Ru", "name":"Ruthenium",       "mass":101.07,    "group":8,  "period":5},
 45:  {"symbol":"Rh", "name":"Rhodium",         "mass":102.90550, "group":9,  "period":5},
 46:  {"symbol":"Pd", "name":"Palladium",       "mass":106.42,    "group":10, "period":5},
 47:  {"symbol":"Ag", "name":"Silver",          "mass":107.8682,  "group":11, "period":5},
 48:  {"symbol":"Cd", "name":"Cadmium",         "mass":112.414,   "group":12, "period":5},
 49:  {"symbol":"In", "name":"Indium",          "mass":114.818,   "group":13, "period":5},
 50:  {"symbol":"Sn", "name":"Tin",             "mass":118.710,   "group":14, "period":5},
 51:  {"symbol":"Sb", "name":"Antimony",        "mass":121.760,   "group":15, "period":5},
 52:  {"symbol":"Te", "name":"Tellurium",       "mass":127.60,    "group":16, "period":5},
 53:  {"symbol":"I",  "name":"Iodine",          "mass":126.90447, "group":17, "period":5},
 54:  {"symbol":"Xe", "name":"Xenon",           "mass":131.293,   "group":18, "period":5},
 55:  {"symbol":"Cs", "name":"Cesium",          "mass":132.90545, "group":1,  "period":6},
 56:  {"symbol":"Ba", "name":"Barium",          "mass":137.327,   "group":2,  "period":6},
 57:  {"symbol":"La", "name":"Lanthanum",       "mass":138.90547, "group":3,  "period":6},
 58:  {"symbol":"Ce", "name":"Cerium",          "mass":140.116,   "group":3,  "period":6},
 59:  {"symbol":"Pr", "name":"Praseodymium",    "mass":140.90766, "group":3,  "period":6},
 60:  {"symbol":"Nd", "name":"Neodymium",       "mass":144.242,   "group":3,  "period":6},
 61:  {"symbol":"Pm", "name":"Promethium",      "mass":145,       "group":3,  "period":6},
 62:  {"symbol":"Sm", "name":"Samarium",        "mass":150.36,    "group":3,  "period":6},
 63:  {"symbol":"Eu", "name":"Europium",        "mass":151.964,   "group":3,  "period":6},
 64:  {"symbol":"Gd", "name":"Gadolinium",      "mass":157.25,    "group":3,  "period":6},
 65:  {"symbol":"Tb", "name":"Terbium",         "mass":158.92535, "group":3,  "period":6},
 66:  {"symbol":"Dy", "name":"Dysprosium",      "mass":162.500,   "group":3,  "period":6},
 67:  {"symbol":"Ho", "name":"Holmium",         "mass":164.93033, "group":3,  "period":6},
 68:  {"symbol":"Er", "name":"Erbium",          "mass":167.259,   "group":3,  "period":6},
 69:  {"symbol":"Tm", "name":"Thulium",         "mass":168.93422, "group":3,  "period":6},
 70:  {"symbol":"Yb", "name":"Ytterbium",       "mass":173.045,   "group":3,  "period":6},
 71:  {"symbol":"Lu", "name":"Lutetium",        "mass":174.9668,  "group":3,  "period":6},
 72:  {"symbol":"Hf", "name":"Hafnium",         "mass":178.49,    "group":4,  "period":6},
 73:  {"symbol":"Ta", "name":"Tantalum",        "mass":180.94788, "group":5,  "period":6},
 74:  {"symbol":"W",  "name":"Tungsten",        "mass":183.84,    "group":6,  "period":6},
 75:  {"symbol":"Re", "name":"Rhenium",         "mass":186.207,   "group":7,  "period":6},
 76:  {"symbol":"Os", "name":"Osmium",          "mass":190.23,    "group":8,  "period":6},
 77:  {"symbol":"Ir", "name":"Iridium",         "mass":192.217,   "group":9,  "period":6},
 78:  {"symbol":"Pt", "name":"Platinum",        "mass":195.084,   "group":10, "period":6},
 79:  {"symbol":"Au", "name":"Gold",            "mass":196.966569,"group":11, "period":6},
 80:  {"symbol":"Hg", "name":"Mercury",         "mass":200.592,   "group":12, "period":6},
 81:  {"symbol":"Tl", "name":"Thallium",        "mass":204.38,    "group":13, "period":6},
 82:  {"symbol":"Pb", "name":"Lead",            "mass":207.2,     "group":14, "period":6},
 83:  {"symbol":"Bi", "name":"Bismuth",         "mass":208.9804,  "group":15, "period":6},
 84:  {"symbol":"Po", "name":"Polonium",        "mass":209,       "group":16, "period":6},
 85:  {"symbol":"At", "name":"Astatine",        "mass":210,       "group":17, "period":6},
 86:  {"symbol":"Rn", "name":"Radon",           "mass":222,       "group":18, "period":6},
 87:  {"symbol":"Fr", "name":"Francium",        "mass":223,       "group":1,  "period":7},
 88:  {"symbol":"Ra", "name":"Radium",          "mass":226,       "group":2,  "period":7},
 89:  {"symbol":"Ac", "name":"Actinium",        "mass":227,       "group":3,  "period":7},
 90:  {"symbol":"Th", "name":"Thorium",         "mass":232.0377,  "group":3,  "period":7},
 91:  {"symbol":"Pa", "name":"Protactinium",    "mass":231.0359,  "group":3,  "period":7},
 92:  {"symbol":"U",  "name":"Uranium",         "mass":238.02891, "group":3,  "period":7},
 93:  {"symbol":"Np", "name":"Neptunium",       "mass":237,       "group":3,  "period":7},
 94:  {"symbol":"Pu", "name":"Plutonium",       "mass":244,       "group":3,  "period":7},
 95:  {"symbol":"Am", "name":"Americium",       "mass":243,       "group":3,  "period":7},
 96:  {"symbol":"Cm", "name":"Curium",          "mass":247,       "group":3,  "period":7},
 97:  {"symbol":"Bk", "name":"Berkelium",       "mass":247,       "group":3,  "period":7},
 98:  {"symbol":"Cf", "name":"Californium",     "mass":251,       "group":3,  "period":7},
 99:  {"symbol":"Es", "name":"Einsteinium",     "mass":252,       "group":3,  "period":7},
100:  {"symbol":"Fm", "name":"Fermium",         "mass":257,       "group":3,  "period":7},
101:  {"symbol":"Md", "name":"Mendelevium",     "mass":258,       "group":3,  "period":7},
102:  {"symbol":"No", "name":"Nobelium",        "mass":259,       "group":3,  "period":7},
103:  {"symbol":"Lr", "name":"Lawrencium",      "mass":266,       "group":3,  "period":7},
104:  {"symbol":"Rf", "name":"Rutherfordium",   "mass":267,       "group":4,  "period":7},
105:  {"symbol":"Db", "name":"Dubnium",         "mass":268,       "group":5,  "period":7},
106:  {"symbol":"Sg", "name":"Seaborgium",      "mass":269,       "group":6,  "period":7},
107:  {"symbol":"Bh", "name":"Bohrium",         "mass":270,       "group":7,  "period":7},
108:  {"symbol":"Hs", "name":"Hassium",         "mass":270,       "group":8,  "period":7},
109:  {"symbol":"Mt", "name":"Meitnerium",      "mass":278,       "group":9,  "period":7},
110:  {"symbol":"Ds", "name":"Darmstadtium",    "mass":281,       "group":10, "period":7},
111:  {"symbol":"Rg", "name":"Roentgenium",     "mass":282,       "group":11, "period":7},
112:  {"symbol":"Cn", "name":"Copernicium",     "mass":285,       "group":12, "period":7},
113:  {"symbol":"Nh", "name":"Nihonium",        "mass":286,       "group":13, "period":7},
114:  {"symbol":"Fl", "name":"Flerovium",       "mass":289,       "group":14, "period":7},
115:  {"symbol":"Mc", "name":"Moscovium",       "mass":290,       "group":15, "period":7},
116:  {"symbol":"Lv", "name":"Livermorium",     "mass":293,       "group":16, "period":7},
117:  {"symbol":"Ts", "name":"Tennessine",      "mass":294,       "group":17, "period":7},
118:  {"symbol":"Og", "name":"Oganesson",       "mass":294,       "group":18, "period":7},
}

# -------------------- UI & LAYOUT PARAMETERS --------------------
LEFT = 40
TOP = 100
# We'll compute cell size so the 18 columns fit comfortable in screen width
# leave margins for UI: left/right ~ 60 each
usable_w = SCREEN_W - (LEFT * 2)
cell_w = int((usable_w - (18 - 1) * 6) / 18)  # gap 6 px
cell_h = 64  # comfortable height
GAP_X = 6
GAP_Y = 10

# More-features overlay (10 features)
FEATURES_LIST = [
    "Molar Mass Calculator",
    "pH Slider",
    "Titration Demo",
    "Reaction Balancer (stub)",
    "Safety Checklist",
    "Lab Notebook",
    "Search Element (keyboard)",
    "Highlight Group (hover)",
    "Period Viewer",
    "Quick Quiz (element)"
]

# Lab notebook memory (simple list)
LAB_NOTES = []

# -------------------- HELPERS --------------------
def draw_centered_text(frame, text, center, size=1.0, color=(230,240,255), thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)
    x = int(center[0] - tw / 2)
    y = int(center[1] + th / 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)

def clamp(v, a, b):
    return max(a, min(b, v))

# -------------------- FEATURES (simple implementations) --------------------
def feature_molar_mass_keyboard(frame):
    # We'll switch to a simple text-mode input using the console.
    # For full GUI keyboard we'd need a significant extra implementation.
    print("Molar Mass Calculator (console input). Enter formula (e.g., H2O) or empty to cancel:")
    formula = input("Formula: ").strip()
    if not formula:
        print("Cancelled.")
        return
    try:
        mm = compute_molar_mass(formula)
        print(f"Molar mass of {formula} ≈ {mm:.4f} g/mol")
    except Exception as e:
        print("Error:", e)

def compute_molar_mass(formula):
    # simple regex parser used by this module
    import re
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    if not parts:
        raise ValueError("Can't parse formula")
    total = 0.0
    for sym, cnt in parts:
        cnt = int(cnt) if cnt else 1
        # find element by symbol
        found = None
        for Z, el in ELEMENTS.items():
            if el["symbol"].lower() == sym.lower():
                found = el["mass"]; break
        if found is None:
            raise ValueError(f"Element '{sym}' not found in table")
        total += found * cnt
    return total

def feature_ph_slider_live(frame, ix, iy):
    # display a horizontal bar and compute pH from finger x position
    w = frame.shape[1]
    x1, x2 = 200, w - 200
    y = 150
    cv2.line(frame, (x1, y), (x2, y), (120, 120, 140), 6)
    pos = clamp(ix, x1, x2) if ix is not None else x1
    conc = (pos - x1) / (x2 - x1) * 1.0  # 0..1 M
    conc = max(1e-6, conc)
    ph = -math.log10(conc)
    cv2.circle(frame, (int(pos), y), 12, (0, 200, 255), -1)
    draw_text(frame, f"Concentration ≈ {conc:.4f} M — pH ≈ {ph:.2f}", (x1, y+30), size=0.8)

def feature_titration_demo(frame, t):
    # simple animated titration paper: just show pH over time simulated
    draw_text(frame, "Titration Demo (visual)", (60, 140))
    # draw a rising curve
    w = frame.shape[1]
    h = frame.shape[0]
    base_x = 200
    len_px = 600
    points = []
    steps = 150
    for i in range(steps):
        x = base_x + int(len_px * i / steps)
        # synthetic pH curve shape
        ph = 2 + 12 * (1 / (1 + math.exp(-(i - steps/2)/6)))
        y = int(h/2 - (ph - 7) * 18)
        points.append((x, y))
    for i in range(len(points)-1):
        cv2.line(frame, points[i], points[i+1], (100,200,255), 2)
    draw_text(frame, "X-axis: added titrant  — Y-axis: pH", (base_x, int(h/2)+120), size=0.6)

def feature_reaction_balancer_stub(frame):
    draw_centered_text(frame, "Reaction Balancer — coming soon", (frame.shape[1]//2, 220), size=0.9)

def feature_safety_checklist(frame):
    tips = [
        "Wear goggles and gloves",
        "Keep chemicals labeled",
        "Know fire extinguisher & eyewash",
        "Do not taste chemicals",
        "Use fume hood for volatile reagents",
    ]
    x, y = 80, 160
    draw_text(frame, "Safety Checklist:", (x, y))
    for i, t in enumerate(tips):
        draw_text(frame, f"- {t}", (x+20, y + 40 + i*34), size=0.7)

def feature_lab_notebook_console():
    print("\nLab Notebook — simple console interface")
    print("1) Add note  2) View notes  (Enter to cancel)")
    c = input("Choice: ").strip()
    if c == "1":
        t = input("Title: ").strip()
        n = input("Note: ").strip()
        LAB_NOTES.append((time.ctime(), t, n))
        print("Saved.")
    elif c == "2":
        for i, r in enumerate(LAB_NOTES, 1):
            print(i, r)
    else:
        print("Cancelled.")

def feature_search_element_console():
    q = input("Search element by name or symbol (case-insensitive): ").strip().lower()
    if not q:
        return
    found = []
    for Z, el in ELEMENTS.items():
        if q == el["symbol"].lower() or q in el["name"].lower():
            found.append((Z, el))
    if not found:
        print("No matches.")
        return
    for Z, el in found:
        print(f"{Z}: {el['name']} ({el['symbol']}) — mass {el['mass']}")

def feature_highlight_group(frame, group):
    # highlight cells of a group (group 1..18)
    for Z, el in ELEMENTS.items():
        if el["group"] == group:
            x1 = LEFT + (el["group"] - 1) * (cell_w + GAP_X)
            y1 = TOP  + (el["period"] - 1) * (cell_h + GAP_Y)
            x2, y2 = x1 + cell_w, y1 + cell_h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 160, 200), 3)

def feature_period_viewer(frame, period):
    # show all elements in that period in a horizontal bar
    elist = [(Z, el) for Z, el in ELEMENTS.items() if el["period"] == period]
    x = 120; y = frame.shape[0] - 220
    draw_text(frame, f"Period {period} elements:", (x, y))
    for i, (Z, el) in enumerate(sorted(elist, key=lambda x: x[0])):
        draw_text(frame, f"{el['symbol']} ({Z})", (x + i*120, y + 40), size=0.7)

def feature_quick_quiz(frame):
    # choose a random element and ask to hover it (console-based answer)
    print("hi")
    return
# Map features to functions (some require console interaction)
FEATURES_FUNCS = {
    0: ("Molar Mass Calculator (console)", feature_molar_mass_keyboard),
    1: ("pH Slider (live)", feature_ph_slider_live),
    2: ("Titration Demo (visual)", feature_titration_demo),
    3: ("Reaction Balancer (stub)", feature_reaction_balancer_stub),
    4: ("Safety Checklist", feature_safety_checklist),
    5: ("Lab Notebook (console)", feature_lab_notebook_console),
    6: ("Search Element (console)", feature_search_element_console),
    7: ("Highlight Group (hover)", feature_highlight_group),
    8: ("Period Viewer", feature_period_viewer),
    9: ("Quick Quiz (console)", feature_quick_quiz),
}

# -------------------- PERIODIC TABLE UI --------------------
def run_periodic_table():
    """
    Main periodic table screen.
    - Full 1600x900 window.
    - Hover with index fingertip: shows tooltip (name + Z).
    - Momentary detail card: hold pinch (thumb+index close) while hovering to show the big detail card.
        Release pinch -> card closes immediately.
    - Press 'm' to toggle features overlay; use keyboard for console-based features.
    - Press ESC or 'b' to exit.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_H)

    last_action_time = 0
    debounce = 0.35  # small debounce for certain feature toggles
    show_features_overlay = False
    running_feature = None  # index of feature that's running (if visual)
    hovered_atomic = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        ix = iy = tx = ty = None
        pinch = False

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            ix, iy = int(hand.landmark[8].x * w), int(hand.landmark[8].y * h)  # index tip
            tx, ty = int(hand.landmark[4].x * w), int(hand.landmark[4].y * h)  # thumb tip

            # fingertip markers
            cv2.circle(frame, (ix, iy), 12, (0, 0, 255), -1)   # red index
            cv2.circle(frame, (tx, ty), 10, (255, 0, 0), -1)   # blue thumb

            d = ((ix - tx)**2 + (iy - ty)**2) ** 0.5
            if d < PINCH_THRESHOLD:
                pinch = True

        # title/header
        cv2.rectangle(frame, (0, 0), (w, 72), (18, 24, 32), -1)
        draw_text(frame, "Periodic Table — hover with index fingertip. Hold pinch to see details.", (24, 44), size=0.8, color=(210, 240, 255))

        # Draw table cells
        hover = None
        for Z, el in ELEMENTS.items():
            g = el["group"]
            p = el["period"]
            x1 = LEFT + (g - 1) * (cell_w + GAP_X)
            y1 = TOP  + (p - 1) * (cell_h + GAP_Y)
            x2, y2 = x1 + cell_w, y1 + cell_h

            # draw cell background & border
            cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 80, 100), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 130, 160), 1)

            # show symbol
            draw_text(frame, el["symbol"], (x1 + 6, y1 + 34), size=0.7, color=(240, 245, 250))

            # hover detection
            if ix is not None and x1 <= ix <= x2 and y1 <= iy <= y2:
                hover = Z
                # highlight cell
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 160), 3)
                # small tooltip
                popup_text = f"{el['name']} ({Z})"
                px = clamp(x1, 8, w - 220)
                py = y1 - 12
                cv2.rectangle(frame, (px - 6, py - 28), (px + 220, py + 6), (22, 28, 36), -1)
                draw_text(frame, popup_text, (px, py - 4), size=0.55, color=(230, 240, 255))

        # momentary detail card: show while pinch is held over a hovered element
        if pinch and hover is not None:
            el = ELEMENTS[hover]
            cw, ch = 700, 360
            cx, cy = w // 2, h // 2
            x1c, y1c = cx - cw // 2, cy - ch // 2
            x2c, y2c = x1c + cw, y1c + ch
            # translucent card background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1c, y1c), (x2c, y2c), (8, 16, 24), -1)
            cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)
            # border & content
            cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (0, 190, 220), 3)
            # big symbol
            draw_text(frame, el["symbol"], (x1c + 40, y1c + 140), size=4.0, color=(245,245,245))
            # details
            draw_text(frame, f"{el['name']} — Atomic Number {hover}", (x1c + 260, y1c + 80), size=0.9, color=(230,240,250))
            draw_text(frame, f"Atomic Mass: {el['mass']}", (x1c + 260, y1c + 140), size=0.8)
            draw_text(frame, f"Group: {el['group']}    Period: {el['period']}", (x1c + 260, y1c + 190), size=0.8)
            draw_text(frame, "Hold pinch to keep card visible. Release pinch to close.", (x1c + 40, y2c - 40), size=0.6, color=(180,190,200))

        # features overlay toggle / UI (press 'm' to toggle)
        # Also draw a small "Features" button top-right that can be hovered/pinched
        fb_w, fb_h = 220, 48
        fb_x1, fb_y1 = w - fb_w - 24, 16
        fb_x2, fb_y2 = fb_x1 + fb_w, fb_y1 + fb_h
        cv2.rectangle(frame, (fb_x1, fb_y1), (fb_x2, fb_y2), (40,40,60), -1)
        cv2.rectangle(frame, (fb_x1, fb_y1), (fb_x2, fb_y2), (80,120,160), 2)
        draw_centered_text(frame, "Features (m)", ((fb_x1 + fb_x2)//2, (fb_y1 + fb_y2)//2 + 6), size=0.7)

        # open features overlay if toggled
        if show_features_overlay:
            # semi-transparent panel on left
            overlay = frame.copy()
            panel_w = 420
            cv2.rectangle(overlay, (24, 96), (24 + panel_w, 96 + 560), (16, 22, 30), -1)
            cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
            cv2.rectangle(frame, (24, 96), (24 + panel_w, 96 + 560), (0, 150, 180), 2)
            draw_text(frame, "Features (press number or use console)", (40, 120), size=0.8)
            for i in range(len(FEATURES_LIST)):
                y = 160 + i * 44
                draw_text(frame, f"{i+1}. {FEATURES_LIST[i]}", (48, y), size=0.6)

        # detect hover+pinch on Features button to toggle
        if ix is not None and fb_x1 <= ix <= fb_x2 and fb_y1 <= iy <= fb_y2 and pinch and (time.time() - last_action_time) > debounce:
            show_features_overlay = not show_features_overlay
            last_action_time = time.time()

        # run visual feature if requested (keyboard-driven)
        if running_feature is not None:
            # call visual features here (indexes 1 or 2 for live)
            if running_feature == 1:
                # pH slider live (use current ix position)
                feature_ph_slider_live(frame, ix if ix is not None else (LEFT + 20), iy)
            elif running_feature == 2:
                feature_titration_demo(frame, time.time())
            elif running_feature == 3:
                feature_reaction_balancer_stub(frame)
            elif running_feature == 4:
                feature_safety_checklist(frame)
            elif running_feature == 7:
                # highlight a sample group (group 1 for demo)
                feature_highlight_group(frame, 1)
            elif running_feature == 8:
                feature_period_viewer(frame, 2)
            elif running_feature == 9:
                feature_quick_quiz(frame)

            draw_text(frame, "Press 'c' to close running feature", (40, h - 40), size=0.6)

        # hints
        draw_text(frame, "ESC/b: Back   |   m: Features overlay   |   Pinch (hold) for detail card", (24, h - 24), size=0.6, color=(180,190,200), thickness=1)

        # show frame
        cv2.imshow("Periodic Table", frame)
        key = cv2.waitKey(1) & 0xFF

        # Keyboard handling (features)
        if key == 27 or key == ord('b'):  # ESC / b to exit periodic table
            break
        if key == ord('m'):
            show_features_overlay = not show_features_overlay
        if key in [ord(str(i)) for i in range(1, 10)] + [ord("0")] and show_features_overlay:

            idx = int(chr(key)) - 1
            if idx in FEATURES_FUNCS:
                # some features are console-based (functions accepting console), others accept (frame, args)
                func = FEATURES_FUNCS[idx][1]
                # visual features indexes for in-frame rendering: 1 (pH),2(titration),3(balancer stub),4(safety),7(highlight),8(period viewer),9(quiz)
                if idx in (1,2,3,4,7,8,9):
                    running_feature = idx
                else:
                    # console features: call directly
                    func(frame)
        if key == ord('c'):
            running_feature = None

    cap.release()
    cv2.destroyAllWindows()

# -------------------- MAIN ENTRY --------------------
def run_chemistry_lab():
    """
    Top-level chemistry lab. Presents a centered menu with three choices:
      - Chemical Reactions
      - Periodic Table (this screen)
      - Reaction Balancer (stub)
      - Back (return)
    Use index fingertip hover + pinch to select menu items.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_H)

    menu_items = ["Chemical Reactions", "Periodic Table", "Reaction Balancer", "Back"]
    selected = -1
    last_action_time = 0
    debounce = 0.35

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        ix = iy = tx = ty = None
        pinch = False
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            ix, iy = int(hand.landmark[8].x * w), int(hand.landmark[8].y * h)
            tx, ty = int(hand.landmark[4].x * w), int(hand.landmark[4].y * h)
            cv2.circle(frame, (ix, iy), 12, (0,0,255), -1)
            cv2.circle(frame, (tx, ty), 10, (255,0,0), -1)
            d = ((ix - tx)**2 + (iy - ty)**2) ** 0.5
            if d < PINCH_THRESHOLD:
                pinch = True

        # draw menu (centered column)
        menu_w = int(w * 0.5)
        menu_h = 80
        start_x = (w - menu_w) // 2
        start_y = 140
        boxes = []
        for i, item in enumerate(menu_items):
            y = start_y + i * (menu_h + 28)
            x1, y1 = start_x, y
            x2, y2 = start_x + menu_w, y + menu_h
            boxes.append((x1, y1, x2, y2))
            # hover effect
            if ix is not None and x1 <= ix <= x2 and y1 <= iy <= y2:
                color = (0, 200, 255)
                selected = i
            else:
                color = (120, 120, 160)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (20,20,28), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            draw_centered_text(frame, item, ((x1+x2)//2 - 60, (y1+y2)//2 + 8), size=0.9, color=color)

        draw_text(frame, "Chemistry Lab", (24, 48), size=1.2, color=(220,240,255))

        # if user pinches on a menu item -> activate (debounced)
        if pinch and ix is not None and (time.time() - last_action_time) > debounce:
            if selected != -1:
                choice = menu_items[selected]
                last_action_time = time.time()
                if choice == "Back":
                    cap.release(); cv2.destroyAllWindows(); return
                elif choice == "Chemical Reactions":
                    # simple reactions screen (animated)
                    run_reactions()
                elif choice == "Periodic Table":
                    run_periodic_table()
                elif choice == "Reaction Balancer":
                    # show placeholder overlay briefly
                    t0 = time.time()
                    while time.time() - t0 < 1.2:
                        ret2, frm2 = cap.read()
                        if not ret2: break
                        frm2 = cv2.flip(frm2, 1)
                        draw_centered_text(frm2, "Reaction Balancer — coming soon", (frm2.shape[1]//2, frm2.shape[0]//2), size=1.0)
                        cv2.imshow("Chemistry Lab", frm2)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break

        draw_text(frame, "Hover with index fingertip. Hold pinch to select. Press ESC to exit.", (24, h-24), size=0.6, color=(190,200,210))
        cv2.imshow("Chemistry Lab", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC exits top-level lab
            break

    cap.release(); cv2.destroyAllWindows()

# If run as main, open the lab directly
if __name__ == "__main__":
    run_chemistry_lab()

# ---------- HELPERS ----------
def compute_button_layout(w, items):
    btn_w, btn_h = int(w * 0.6), 80
    start_x, start_y = (w - btn_w) // 2, 150
    boxes = []
    for i, item in enumerate(items):
        y = start_y + i * (btn_h + 30)
        boxes.append((start_x, y, start_x + btn_w, y + btn_h))
    return boxes


def draw_buttons(frame, items, selected, w):
    boxes = compute_button_layout(w, items)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        scale = 1.1 if i == selected else 1.0
        bx1 = int(cx - (x2 - x1)//2 * scale)
        by1 = int(cy - (y2 - y1)//2 * scale)
        bx2 = int(cx + (x2 - x1)//2 * scale)
        by2 = int(cy + (y2 - y1)//2 * scale)
        color = (0, 200, 255) if i == selected else (100, 100, 180)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (40, 40, 60), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 3)
        draw_text(frame, items[i], (cx - 100, cy + 10), 0.9, color)
    return boxes

# ---------- FEATURES ----------
def run_reactions():
    cap = cv2.VideoCapture(0)
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w, _ = frame.shape
        cv2.putText(frame, "Reaction: 2H2 + O2 -> 2H2O", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if frames % 40 < 20:
            cv2.circle(frame, (w//2-100, h//2), 30, (255, 0, 0), -1)
            cv2.circle(frame, (w//2+100, h//2), 30, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (w//2, h//2), 40, (200, 200, 255), -1)
        cv2.imshow("Chemical Reactions", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
        frames += 1
    cap.release(); cv2.destroyAllWindows()

def run_reaction_balancer():
    print("⚗️ Reaction balancer coming soon...")
    return






# ---------- MAIN ----------
def run_chemistry_lab():
    global chemistry_menu, chemistry_selected

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame and convert to RGB for hand detection
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        finger_x, finger_y, pinch = None, None, False

        # Detect hand landmarks
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            index_x = int(hand.landmark[8].x * w)
            index_y = int(hand.landmark[8].y * h)
            thumb_x = int(hand.landmark[4].x * w)
            thumb_y = int(hand.landmark[4].y * h)

            finger_x, finger_y = index_x, index_y
            distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5
            pinch = distance < PINCH_THRESHOLD

        # Draw menu buttons
        items = menus_chemistry[chemistry_menu]
        boxes = draw_buttons(frame, items, chemistry_selected, w)

        chemistry_selected = -1
        if finger_x and finger_y:
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                if x1 <= finger_x <= x2 and y1 <= finger_y <= y2:
                    chemistry_selected = i
                    if pinch:
                        choice = items[i]
                        if choice == "Back":
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                        elif choice == "Chemical Reactions":
                            run_reactions()
                        elif choice == "Periodic Table":
                            run_periodic_table()
                        elif choice == "Reaction Balancer":
                            run_reaction_balancer()

        cv2.imshow("Chemistry Lab", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()





