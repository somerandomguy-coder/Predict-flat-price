"""Render pitch.mp4 from our results. Edit VALUES or the scenes, then re-run: about a minute.
Run: PYTHONPATH=. .venv/bin/python video/make_pitch.py
"""
import json, subprocess, glob, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
m = json.load(open(ROOT / "out/real/metrics.json"))["folds"]["A"]
VALUES = dict(flats="65,777", with_=1010, without=930, miss=round(m["mae"]), within=round(m["within100"] * 100),
              areas=[("Pyrmont", 70), ("Sydney CBD", 69), ("Surry Hills", 49), ("Waterloo", 4)])  # the report's section 2 table
V = VALUES
gap = V["with_"] - V["without"]
W, H, FPS = 1280, 720, 24
BG, INK, MUTED, BLUE, ORANGE, PANEL = (15, 23, 42), (248, 250, 252), (160, 174, 192), (56, 135, 229), (235, 104, 52), (30, 41, 59)
fonts = glob.glob("/usr/share/fonts/**/DejaVuSans-Bold.ttf", recursive=True) + glob.glob("/usr/share/fonts/**/LiberationSans-Bold.ttf", recursive=True)
FONT = fonts[0] if fonts else subprocess.check_output(["fc-match", "-f", "%{file}", "sans:bold"]).decode()
F = lambda s: ImageFont.truetype(FONT, s)


def ease(t): t = max(0, min(1, t)); return t * t * (3 - 2 * t)
def mix(c, a): return tuple(int(BG[i] + (c[i] - BG[i]) * a) for i in range(3))


def text(d, t, y, size, c=INK, a=1, x=None):
    f = F(size); w = d.textlength(t, font=f)
    d.text(((W - w) / 2 if x is None else x, y + (1 - a) * 18), t, font=f, fill=mix(c, a))


def bar(d, x, y, w, h, c, a=1): 
    if w > 2: d.rounded_rectangle([x, y, x + w, y + h], 8, fill=mix(c, a))


def s_problem(d, t):
    text(d, "No car. Still paying for parking?", 90, 50, INK, ease(t / .8))
    text(d, "A 2-bedroom flat in Pyrmont", 190, 34, MUTED, ease((t - 1) / .8))
    text(d, f"${V['with_']:,} a week", 280, 96, ORANGE, ease((t - 2) / .8))
    text(d, "with a car space bundled in", 410, 32, MUTED, ease((t - 2.6) / .8))
    text(d, "How much of that rent pays for empty concrete?", 520, 36, INK, ease((t - 4.5) / .8))


def s_model(d, t):
    text(d, "Our model prices both, for every flat", 70, 44, INK, ease(t / .7))
    full = 800
    for i, (lab, v, c, st) in enumerate((("With a car space", V["with_"], ORANGE, 1), ("Without", V["without"], BLUE, 2))):
        y = 200 + i * 130; p = ease((t - st) / 1)
        text(d, lab, y - 38, 28, MUTED, ease((t - st) / .5), x=150)
        bar(d, 150, y, full * v / V["with_"] * p, 70, c)
        text(d, f"${v:,}", y + 10, 42, INK, ease((t - st - .7) / .5), x=985)
    text(d, f"The space costs about ${gap} a week", 480, 46, ORANGE, ease((t - 3.6) / .7))
    text(d, f"That is ${gap * 52:,} a year", 550, 40, INK, ease((t - 4.6) / .7))
    text(d, "Typical Pyrmont 2-bedroom, our estimate as at 31 July 2026", 640, 22, MUTED, ease((t - 4.6) / .7))


def s_win(d, t):
    text(d, "Now both sides can do better", 70, 46, INK, ease(t / .7))
    for x, head, c, lines, st in ((90, "Tenant", BLUE, ["No car? Ask for the flat", "without the space.", f"Pay ${V['without']}, not ${V['with_']:,}."], 1),
                                  (670, "Owner", ORANGE, ["Lease the space separately", f"for about ${gap} a week,", "to someone who needs it."], 3.2)):
        a = ease((t - st) / .7)
        d.rounded_rectangle([x, 170, x + 520, 520], 18, fill=mix(PANEL, a))
        text(d, head, 200, 44, c, a, x=x + 34)
        for i, l in enumerate(lines): text(d, l, 290 + i * 62, 31, INK, ease((t - st - .5 - i * .3) / .6), x=x + 34)
    text(d, "A fair number to start the conversation", 590, 34, MUTED, ease((t - 5.8) / .7))


def s_trust(d, t):
    text(d, "Why trust the number?", 60, 46, INK, ease(t / .7))
    text(d, "Tested on 2,635 later adverts it never saw", 135, 28, MUTED, ease((t - .5) / .7))
    for i, (lab, v, c, st) in enumerate((("Simple estimate", 100, MUTED, 1), ("Our model", V["miss"], BLUE, 2.2))):
        y = 245 + i * 115; p = ease((t - st) / 1)
        text(d, lab, y - 36, 26, MUTED, ease((t - st) / .5), x=190)
        bar(d, 190, y, 6.6 * v * p, 62, c)
        text(d, f"${v} typical miss", y + 12, 30, INK, ease((t - st - .7) / .5), x=210 + 6.6 * v * p)
    text(d, f"{V['within']}% landed within $100 a week", 505, 46, ORANGE, ease((t - 3.8) / .7))
    text(d, "Biggest clue: the flat's own last rent, brought up to date", 585, 30, INK, ease((t - 5) / .7))
    text(d, "Simple estimate and our model use held-back adverts. Microburbs reports $86 on different adverts.", 650, 20, MUTED, ease((t - 5) / .7))


def s_area(d, t):
    text(d, "A car space is not worth the same everywhere", 60, 42, INK, ease(t / .7))
    mx = max(v for _, v in V["areas"])
    for i, (n, v) in enumerate(V["areas"]):
        y = 165 + i * 88; st = .8 + i * .45; p = ease((t - st) / .8)
        text(d, n, y + 8, 30, INK, ease((t - st) / .5), x=130)
        bar(d, 400, y, 620 * v / mx * p, 54, ORANGE if v >= 40 else BLUE)
        text(d, f"${v}", y + 8, 32, INK, ease((t - st - .6) / .5), x=420 + 620 * v / mx * p)
    text(d, "Estimated $ a week, from section 2 of our report. All flat sizes, within-building comparisons.", 515, 20, MUTED, ease((t - 2) / .6))
    text(d, "About $70 in Pyrmont, about $4 in Waterloo", 575, 38, ORANGE, ease((t - 4) / .7))


def s_close(d, t):
    text(d, "Know what you are paying for.", 215, 58, INK, ease(t / .8))
    text(d, f"{V['flats']} flats. Two prices each.", 320, 40, MUTED, ease((t - 1.3) / .8))
    text(d, "2 Bulbs 1 Source", 455, 44, ORANGE, ease((t - 2.6) / .8))
    text(d, "Microburbs Hackathon, 7 November 2026", 525, 26, MUTED, ease((t - 3) / .8))


SCENES = [(s_problem, 8), (s_model, 9), (s_win, 9), (s_trust, 9), (s_area, 8), (s_close, 6)]
clips = [ROOT / f"video/audio/scene_{i}.mp3" for i in range(len(SCENES))]
VOICE = all(c.exists() for c in clips)
if VOICE:  # each scene lasts as long as its narration, plus a short pause
    length = lambda c: float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(c)]))
    lens = [length(c) for c in clips]
    BASE = {f: d for f, d in SCENES}
    SCENES = [(f, l + .35 + (.7 if i == len(SCENES) - 1 else 0)) for i, ((f, d), l) in enumerate(zip(SCENES, lens))]
out = ROOT / "video/audio/silent.mp4" if VOICE else ROOT / "pitch.mp4"
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                       "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                       "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
for fn, dur in SCENES:
    for i in range(int(dur * FPS)):
        t = i / FPS
        im = Image.new("RGB", (W, H), BG); fn(ImageDraw.Draw(im), t * (BASE[fn] / dur if VOICE else 1))  # fit the animation to the voice
        if t > dur - .4:
            im = Image.blend(im, Image.new("RGB", (W, H), BG), (t - dur + .4) / .4)
        ff.stdin.write(im.tobytes())
ff.stdin.close(); ff.wait()
if VOICE:
    start, ins, fl = 0.0, [], []
    for i, (c, (_, d)) in enumerate(zip(clips, SCENES)):
        ins += ["-i", str(c)]; fl.append(f"[{i + 1}:a]adelay={int((start + .25) * 1000)}:all=1[a{i}]"); start += d
    mixf = ";".join(fl) + ";" + "".join(f"[a{i}]" for i in range(len(clips))) + f"amix=inputs={len(clips)}:normalize=0[a]"
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), *ins, "-filter_complex", mixf, "-map", "0:v", "-map", "[a]",
                           "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", str(start), "-movflags", "+faststart", str(ROOT / "pitch.mp4")])
print("wrote pitch.mp4", round(sum(d for _, d in SCENES), 1), "s, voice:", VOICE)
