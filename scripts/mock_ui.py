#!/usr/bin/env python3
"""
Mock-UI renderer for computer-use data (Phase 6 prep).

Draws synthetic B&W desktop UIs with PIL (provided via `uv run --with
pillow`), samples L1-L3 tasks, and emits (screenshot, action) sessions
with AUTO-GRADING (the renderer knows every bbox and state, so success
criteria are code, not labels -- the same renderer later serves as the
RL verifier).

Canvas 480x320 grayscale. Action space (pixel-grounded, vision must
learn coordinates):
    click X Y        -- mouse click at pixels
    type "TEXT"      -- type into the focused field (must click field first)
Screens re-render after every state-changing action.

Levels:
    L1 ground  "Click the {label} button." -> click in bbox.
    L2 toggle  "Turn {label} on/off."       -> click checkbox, state flips.
    L3 form    "Fill Name with '{v}' and submit." -> click field, type,
               click submit; success banner shows the value.

Output: ~/.cache/autoresearch/mock_ui/sNNNN/
    screen_K.png, transcript.jsonl ({screen, action, grade, detail})

Usage:
    uv run --with pillow scripts/mock_ui.py --n 5 --seed 7
"""

import argparse
import json
import os
import random
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("need PIL: uv run --with pillow scripts/mock_ui.py ...")

W, H = 480, 320
CACHE = os.path.expanduser("~/.cache/autoresearch")
FONT = ImageFont.load_default()

BUTTON_LABELS = ["OK", "Cancel", "Submit", "Save", "Open", "Close", "Apply",
                 "Retry", "Delete", "Search", "Login", "Send"]
FIELD_LABELS = ["Name", "Email", "City", "Phone", "Title"]
CHECK_LABELS = ["Remember me", "Dark mode", "Auto-save", "Notify", "Backup"]
NAMES = ["Ada", "bob42", "Linz", "Zoe", "Max", "lena", "Oslo", "Kim"]


class UI:
    """One mock window: elements, state, rasterizer, stepper."""

    def __init__(self, rng):
        self.rng = rng
        self.title = "Settings" if rng.random() < 0.5 else "Login"
        self.buttons = []   # (label, box)
        self.fields = []    # {label, box, value, focused}
        self.checks = []    # {label, box, on}
        self.banner = ""
        self._layout()

    def _layout(self):
        r = self.rng
        # 2-4 buttons along the bottom
        labels = r.sample(BUTTON_LABELS, r.randint(2, 4))
        x = 20
        for lab in labels:
            w = 12 * len(lab) + 20
            self.buttons.append((lab, (x, H - 52, x + w, H - 24)))
            x += w + 12
        # 1-2 text fields
        y = 60
        for lab in r.sample(FIELD_LABELS, r.randint(1, 2)):
            self.fields.append({"label": lab, "box": (20, y, 300, y + 24),
                                "value": "", "focused": False})
            y += 40
        # 1-2 checkboxes
        for lab in r.sample(CHECK_LABELS, r.randint(1, 2)):
            self.checks.append({"label": lab, "box": (20, y, 36, y + 16),
                                "on": r.random() < 0.5})
            y += 26

    # -- raster ---------------------------------------------------------
    def render(self):
        img = Image.new("L", (W, H), 255)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W - 1, 23], fill=0)
        d.text((8, 6), self.title, font=FONT, fill=255)
        for lab, (x0, y0, x1, y1) in self.buttons:
            d.rectangle([x0, y0, x1, y1], outline=0, width=2)
            d.text((x0 + 10, y0 + 6), lab, font=FONT, fill=0)
        for f in self.fields:
            x0, y0, x1, y1 = f["box"]
            d.text((x0, y0 - 14), f["label"] + ":", font=FONT, fill=0)
            d.rectangle([x0, y0, x1, y1], outline=0, width=1)
            d.text((x0 + 4, y0 + 5),
                   f["value"] + ("|" if f["focused"] else ""),
                   font=FONT, fill=0)
        for c in self.checks:
            x0, y0, x1, y1 = c["box"]
            d.rectangle([x0, y0, x1, y1], outline=0, width=2)
            if c["on"]:
                d.line([x0 + 3, y0 + 8, x0 + 7, y0 + 13,
                        x1 - 3, y0 + 3], fill=0, width=2)
            d.text((x1 + 8, y0 + 2), c["label"], font=FONT, fill=0)
        if self.banner:
            d.rectangle([20, H - 80, W - 20, H - 60], outline=0, width=1)
            d.text((28, H - 74), self.banner, font=FONT, fill=0)
        return img

    # -- interaction ----------------------------------------------------
    @staticmethod
    def _hit(box, x, y):
        x0, y0, x1, y1 = box
        return x0 <= x <= x1 and y0 <= y <= y1

    def click(self, x, y):
        """Apply click; returns (changed: bool, what: str)."""
        for f in self.fields:
            f["focused"] = self._hit(f["box"], x, y)
        for c in self.checks:
            if self._hit(c["box"], x, y):
                c["on"] = not c["on"]
                return True, f"toggle {c['label']} -> {c['on']}"
        for lab, box in self.buttons:
            if self._hit(box, x, y):
                if lab in ("Submit", "Login", "Send", "Save"):
                    vals = " ".join(f["value"] for f in self.fields)
                    self.banner = f"Submitted: {vals}".strip() or "Submitted"
                return True, f"press {lab}"
        return False, "background"

    def type(self, text):
        for f in self.fields:
            if f["focused"]:
                f["value"] += text
                return True
        return False


# -- tasks ---------------------------------------------------------------
def task_l1(ui, rng):
    lab, box = rng.choice(ui.buttons)
    x0, y0, x1, y1 = box
    return (f"Click the {lab} button.",
            f"click {(x0 + x1) // 2} {(y0 + y1) // 2}",
            ("inbox", box))


def task_l2(ui, rng):
    c = rng.choice(ui.checks)
    want = not c["on"]
    x0, y0, x1, y1 = c["box"]
    return (f"Turn {c['label']} {'on' if want else 'off'}.",
            f"click {(x0 + x1) // 2} {(y0 + y1) // 2}",
            ("checkbox", c["label"], want))


def task_l3(ui, rng):
    f = rng.choice(ui.fields)
    val = rng.choice(NAMES)
    subs = [b for b, _ in ui.buttons
            if b in ("Submit", "Login", "Send", "Save")]
    sub = subs[0] if subs else ui.buttons[0][0]
    tracks = sub in ("Submit", "Login", "Send", "Save")
    fx0, fy0, fx1, fy1 = f["box"]
    sx0, sy0, sx1, sy1 = dict(ui.buttons)[sub]
    steps = [f"click {(fx0 + fx1) // 2} {(fy0 + fy1) // 2}",
             f'type "{val}"',
             f"click {(sx0 + sx1) // 2} {(sy0 + sy1) // 2}"]
    return (f"Fill {f['label']} with '{val}' and press {sub}.",
            steps, ("form", f["label"], val, tracks))


def grade(ui, rule, action):
    kind = rule[0]
    if kind == "inbox":
        parts = action.split()
        return UI._hit(rule[1], int(parts[1]), int(parts[2]))
    if kind == "checkbox":
        _, lab, want = rule
        return next(c["on"] for c in ui.checks if c["label"] == lab) == want
    if kind == "form":
        _, lab, val, tracks = rule
        f = next(ff for ff in ui.fields if ff["label"] == lab)
        if f["value"] != val:
            return False
        return (val in ui.banner) if tracks else True
    return False


def run_session(sid, level, seed, outdir):
    rng = random.Random(seed)
    ui = UI(rng)
    sdir = os.path.join(outdir, f"s{sid:04d}")
    os.makedirs(sdir, exist_ok=True)
    if level == 1:
        inst, act, rule = task_l1(ui, rng)
        steps = [act]
    elif level == 2:
        inst, act, rule = task_l2(ui, rng)
        steps = [act]
    else:
        inst, steps, rule = task_l3(ui, rng)
    trans = [{"instruction": inst, "level": level}]
    shot = 0
    ui.render().save(os.path.join(sdir, f"screen_{shot}.png"))
    ok = True
    for act in steps:
        parts = act.split(None, 1)
        if parts[0] == "click":
            x, y = map(int, parts[1].split())
            ui.click(x, y)
            good = True  # graded at end for L1 via rule
        elif parts[0] == "type":
            ui.type(parts[1].strip('"'))
            good = True
        else:
            good = False
        shot += 1
        ui.render().save(os.path.join(sdir, f"screen_{shot}.png"))
        trans.append({"screen": shot - 1, "action": act, "ok_step": good})
        ok = ok and good
    final = grade(ui, rule, steps[-1] if level == 1 else "")
    trans.append({"final_grade": final, "rule": list(rule)})
    with open(os.path.join(sdir, "transcript.jsonl"), "w") as f:
        for t in trans:
            f.write(json.dumps(t) + "\n")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(CACHE, "mock_ui"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    results = []
    for i in range(args.n):
        level = rng.choice([1, 1, 2, 3])
        good = run_session(i, level, args.seed * 1000 + i, args.out)
        results.append((level, good))
    n_ok = sum(1 for _, g in results if g)
    print(f"sessions: {len(results)}, perfect-play grades pass: {n_ok}")
    # negative control: wrong click must FAIL L1 grade
    ui = UI(random.Random(0))
    _, _, rule = task_l1(ui, random.Random(1))
    print("negative control (click 5,5):",
          "FAIL-ok" if not grade(ui, rule, "click 5 5") else "BROKEN")
    if n_ok != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
