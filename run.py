# run.py — Viral Shorts Generator (local, modular, fast)
# - yt-dlp download or local file
# - faster-whisper transcription (word-level)
# - AI clip selection via Ollama (deepseek-r1:8b default), with strict JSON + self-validation
# - Smarter fallback selector (sentiment + Qs + numbers + optional audio energy + proximity)
# - Boundary validation + micro-story checks (Hook→Build→Payoff)
# - Karaoke ASS (adaptive font + power-word emphasis)
# - Portrait pad + render via FFmpeg
# - Context-aware title system (variants + scoring) + hashtags/desc
# - JSONL decision logging for future feedback loops
# --------------------------------------------------------------------

import os, sys, glob, json, math, re, subprocess, tempfile, time
from typing import List, Dict, Any, Optional, Tuple
from faster_whisper import WhisperModel
import pysubs2
import google.generativeai as genai
from google.generativeai import GenerationConfig

# ====================== SHELL UTILITY ======================
def sh(cmd: str):
    print(">>", cmd, flush=True)
    subprocess.check_call(cmd, shell=True)

# ====================== ENV / PATHS ========================
URL = os.environ.get("URL")
FILE = os.environ.get("FILE", "input.mp4")
OUT_DIR = os.environ.get("OUT_DIR", "out")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_SRT  = os.path.join(OUT_DIR, "out.srt")
OUT_JSON = os.path.join(OUT_DIR, "words.json")
OUT_ASS  = os.path.join(OUT_DIR, "karaoke.ass")
OUT_MP4  = os.path.join(OUT_DIR, "short_1080x1920.mp4")
OUT_TITLE = os.path.join(OUT_DIR, "title.txt")
OUT_DESC  = os.path.join(OUT_DIR, "description.txt")
OUT_LOG   = os.path.join(OUT_DIR, "selection_log.jsonl")

MODEL     = os.environ.get("MODEL", "medium")
COMPUTE   = os.environ.get("COMPUTE", "int8")
BEAM      = int(os.environ.get("BEAM_SIZE", "5"))
DEVICE    = os.environ.get("DEVICE", "auto")
TURBO_MODE = os.environ.get("TURBO_MODE", "0") == "1"

# Automatically adjust compute type based on device
try:
    import torch
    if not torch.cuda.is_available():
        if COMPUTE == "float16":
            print("WARNING: CUDA not available, switching to int8.")
            COMPUTE = "int8"
        if DEVICE == "cuda":
            print("WARNING: CUDA not available, switching to CPU.")
            DEVICE = "cpu"
except ImportError:
    print("WARNING: PyTorch not found. Assuming CPU, switching to int8 if float16 is set.")
    if COMPUTE == "float16":
        COMPUTE = "int8"
    DEVICE = "cpu"

FFMPEG_VCODEC = "libx264"
if DEVICE == "cuda":
    FFMPEG_VCODEC = "h264_nvenc"
    print(f"> CUDA detected, using FFmpeg hardware encoder: {FFMPEG_VCODEC}")

_temp_env = os.environ.get("TEMPERATURE")
TEMP: Any
if _temp_env is not None:
    try:
        if "," in _temp_env:
            TEMP = tuple(float(t.strip()) for t in _temp_env.split(','))
        else:
            TEMP = float(_temp_env)
    except ValueError:
        print(f"  WARNING: Invalid TEMPERATURE value '{_temp_env}'. Using default.")
        TEMP = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
else:
    TEMP = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
VAD       = os.environ.get("VAD", "1") == "1"
WORD_TS   = os.environ.get("WORD_TS", "1") == "1"
LANG      = os.environ.get("LANGUAGE")
AI_MODEL  = os.environ.get("AI_MODEL", "gemini-2.0-flash") # Set to the model available in your AI Studio
if AI_MODEL.startswith('='): AI_MODEL = AI_MODEL[1:]

TGT       = float(os.environ.get("TARGET_SEC", "12"))
MIN_LEN   = float(os.environ.get("MIN_SEC", "8"))
MAX_LEN   = float(os.environ.get("MAX_SEC", "15"))

MULTI_SEG = os.environ.get("MULTI_SEGMENT", "1") == "1"
USE_AI    = os.environ.get("USE_AI", "1") == "1"
STRICT_PROX = os.environ.get("STRICT_PROXIMITY", "1") == "1"

AUDIO_ANALYSIS = os.environ.get("AUDIO_ANALYSIS", "0") == "1"
SUBS_ADAPTIVE  = os.environ.get("SUBS_ADAPTIVE_FONTS", "1") == "1"
SUBS_HILITE    = os.environ.get("SUBS_POWER_HILITE", "1") == "1"

TITLE_TOP_K = int(os.environ.get("TITLE_TOP_K", "5"))
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


# ====================== CLEANUP OLD OUTPUTS =================
print("> Cleaning old outputs...")
for p in [OUT_SRT, OUT_JSON, OUT_ASS, OUT_MP4, OUT_TITLE, OUT_DESC]:
    if os.path.exists(p):
        try:
            os.remove(p)
            print(f"  Removed: {p}")
        except Exception as e:
            print(f"  Could not remove {p}: {e}")
for segf in glob.glob(os.path.join(OUT_DIR, "segment_*.mp4")):
    try: os.remove(segf)
    except: pass
for tmp in ["concat_list.txt", "temp_concat.mp4", "temp_karaoke.ass"]:
    p = os.path.join(OUT_DIR, tmp)
    if os.path.exists(p):
        try: os.remove(p)
        except: pass

# ====================== YT-DLP / INPUT =====================
def get_video_metadata(url: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        res = subprocess.run(
            ['yt-dlp', '--print', '%(channel)s|||%(title)s', '--no-playlist', url],
            capture_output=True, text=True, timeout=30
        )
        if res.returncode == 0:
            s = res.stdout.strip()
            if '|||' in s:
                ch, title = s.split('|||', 1)
                return ch.strip(), title.strip()
    except Exception as e:
        print(f"  Could not extract metadata: {e}")
    return None, None

INFILE = FILE

print(f"> Model={MODEL} compute={COMPUTE} beam={BEAM} vad={VAD} word_ts={WORD_TS}")
print(f"> Multi-seg: {MULTI_SEG}  USE_AI: {USE_AI}  Strict Prox: {STRICT_PROX}")
print(f"> Input: {INFILE}")
if os.path.exists(INFILE):
    size_mb = os.path.getsize(INFILE) / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")
    if size_mb == 0:
        print("FATAL: Input file is 0 bytes. Aborting.", file=sys.stderr)
        sys.exit(1)
else:
    print(f"FATAL: Input file '{INFILE}' not found. Aborting.", file=sys.stderr)
    sys.exit(1)

# ====================== TRANSCRIBE =========================
print("> Loading Whisper model...")
model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)

if TURBO_MODE:
    BEAM = 1
    print("> TURBO MODE enabled: Using beam_size=1 for faster transcription.")
print("> Transcribing with word timestamps...")
try:
    segments, _ = model.transcribe(
        INFILE, language=LANG, vad_filter=VAD,
        word_timestamps=WORD_TS, condition_on_previous_text=False,
        temperature=TEMP, beam_size=BEAM
    )
    segments_list = list(segments)
    print(f"  ✓ Transcription complete: {len(segments_list)} segments")
except Exception as e:
    print(f"  ✗ TRANSCRIPTION FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

def fmt_time(x: float) -> str:
    h = int(x // 3600); m = int((x % 3600) // 60); s = (x % 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')

print("> Generating SRT...")
with open(OUT_SRT, "w", encoding="utf-8") as f:
    for i, s in enumerate(segments_list, 1):
        f.write(f"{i}\n{fmt_time(s.start)} --> {fmt_time(s.end)}\n{s.text.strip()}\n\n")
print("OK ->", OUT_SRT)

print("> Extracting word timestamps...")
try:
    if not segments_list:
        print("FATAL: Transcription produced no segments. Aborting.", file=sys.stderr)
        sys.exit(1)

    words: List[Dict[str,Any]] = []
    for seg in segments_list:
        if getattr(seg, "words", None):
            for w in seg.words:
                words.append({
                    "start": float(w.start), "end": float(w.end),
                    "text": w.word, "prob": float(getattr(w, "probability", 0.5) or 0.5)
                })

    print(f"  ✓ Extracted {len(words)} words")
except Exception as e:
    print(f"  ✗ WORD EXTRACTION FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)


with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(words, f, ensure_ascii=False, indent=2)
print("OK ->", OUT_JSON)

if not words:
    print("No words available; aborting.", file=sys.stderr)
    sys.exit(0)

print(f"> Total words: {len(words)}")

# ====================== PRE-COMPUTATION & ANALYSIS =====================
print("\n> Performing pre-computation and analysis...")

# --- Audio Analysis ---
energy_series = None
silences = []
if AUDIO_ANALYSIS:
    print("  Analyzing audio energy...")
    energy_series = extract_energy_peaks(INFILE, enabled=True)
    print("  Detecting silences...")
    silences = detect_silences(INFILE, enabled=True)
    print(f"  Found {len(energy_series) or 0} energy datapoints and {len(silences)} silences.")

# --- Word Filtering ---
initial_word_count = len(words)
words = [w for w in words if w.get('prob', 0) >= 0.3]
filtered_count = initial_word_count - len(words)
if filtered_count > 0:
    print(f"  Filtered out {filtered_count} low-confidence words (prob < 0.3).")


# ====================== ANALYSIS UTILS =====================
POS_WORDS = set("""
amazing awesome incredible shocking insane mind-blowing love happy excited wow best secret reveal truth finally exposed hack trick
""".split())
NEG_WORDS = set("""
hate bad worst mistake wrong fail failed never angry stupid dumb problem issue crisis
""".split())
QUESTION_RE = re.compile(r'\?')
NUMBER_RE   = re.compile(r'\b\d+(\.\d+)?\b')

def sentiment_score(text: str) -> float:
    t = text.lower()
    score = 0.0
    for w in POS_WORDS:
        if w in t: score += 1.0
    for w in NEG_WORDS:
        if w in t: score += 1.0  # strong negative still engages
    score += t.count("!") * 0.5
    for w in ["never","secret","truth","exposed","finally","shocking","mistake","wrong","hack","trick"]:
        score += t.count(w) * 1.2
    score += len(NUMBER_RE.findall(t)) * 0.6
    return score

def question_score(text: str) -> float:
    return 2.5 if QUESTION_RE.search(text) else 0.0

def boundary_ok(start_text: str, end_text: str) -> bool:
    start_ok = bool(start_text and start_text[0].isupper())
    end_ok   = bool(end_text and end_text[-1] in ".!?")
    return start_ok and end_ok

def arc_role(i: int, n: int) -> str:
    if n == 2: return ["hook","payoff"][i]
    return ["hook","buildup","payoff"][i]

def proximity_penalty(prev_end: float, curr_start: float) -> float:
    gap = max(0.0, curr_start - prev_end)
    if gap <= 5.0: return 0.0
    if gap <= 10.0: return (gap-5.0)*0.5
    return 999.0

def extract_energy_peaks(path: Optional[str], step_sec: float = 0.5, enabled: bool = False):
    if not enabled or not path: return None
    try:
        cmd = f'ffmpeg -hide_banner -i "{path}" -af astats=metadata=1:reset=1 -f null -'
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        times, rms = [], []
        t = 0.0
        for line in p.stdout:
            if "RMS_level" in line:
                try:
                    val = float(line.strip().split("RMS_level:")[-1])
                    e = 10 ** (val/20.0)
                    times.append(t); rms.append(e); t += step_sec
                except: pass
        p.wait()
        return list(zip(times, rms))
    except: return None

def energy_in_window(energy_series, start: float, end: float) -> float:
    if not energy_series: return 0.0
    acc = 0.0; n = 0
    for t, e in energy_series:
        if start <= t <= end:
            acc += e; n += 1
    return (acc / max(1, n))

def detect_silences(path: Optional[str], enabled: bool = False) -> List[Dict[str, float]]:
    if not enabled or not path: return []
    try:
        cmd = f'ffmpeg -i "{path}" -af silencedetect=noise=-30dB:d=0.5 -f null -'
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        silences = []
        for line in p.stdout:
            if "silence_start" in line:
                start = float(line.split("silence_start: ")[1])
                silences.append({"start": start})
            elif "silence_end" in line and silences:
                end = float(line.split("silence_end: ")[1].split(" |")[0])
                silences[-1]["end"] = end
        p.wait()
        return [s for s in silences if "end" in s and (s["end"] - s["start"]) > 0.5]
    except Exception as e:
        print(f"  Silence detection failed: {e}")
        return []

# ====================== OLLAMA AI SELECTION =================
def find_dense_regions(segments_list, window_size=30.0, min_segments=4):
    """Find regions of the transcript with high segment density."""
    dense_regions = []

    for i in range(len(segments_list)):
        window_start = segments_list[i].start
        window_end = window_start + window_size

        # Count segments in this window
        segments_in_window = [
            s for s in segments_list
            if s.start >= window_start and s.end <= window_end
        ]

        if len(segments_in_window) >= min_segments:
            dense_regions.append({
                'start': window_start,
                'end': window_end,
                'segment_count': len(segments_in_window),
                'segments': segments_in_window
            })

    return dense_regions

def generate_candidates(segments, silences, min_dur, max_dur, max_gap=20.0, max_candidates=10):
    """Generates valid multi-segment clip candidates based on rules."""
    print("\n> Generating clip candidates...")
    candidates = []

    # --- 1. Generate Single, Extended "Play Out" Segments ---
    for seg in segments:
        base_dur = seg.end - seg.start
        if base_dur > max_dur: continue

        # Find silence immediately following the segment to determine "play out" time
        playout_time = 0.0
        for silence in silences:
            if seg.end < silence['start'] < seg.end + 1.5: # Silence starts shortly after
                playout_time = min(silence['end'] - seg.end, 8.0) # Cap playout at 8s
                break
        
        total_dur = base_dur + playout_time
        if min_dur <= total_dur <= max_dur:
            score = sentiment_score(seg.text) * 1.5 + question_score(seg.text) * 2.0
            candidates.append({
                "score": score,
                "type": "single_extended",
                "segments": [{"start": seg.start, "end": seg.end + playout_time, "text": seg.text.strip(), "role": "hook"}],
                "total_duration": total_dur,
                "playout_duration": playout_time
            })

    # --- 2. Generate Logical Multi-Segment Clips ---
    for i in range(len(segments) - 1):
        seg1 = segments[i]
        # Only look at the next few segments to ensure logical continuation
        for j in range(i + 1, min(i + 5, len(segments))):
            seg2 = segments[j]
            gap = seg2.start - seg1.end
            if 0 <= gap <= max_gap:
                total_dur = (seg1.end - seg1.start) + (seg2.end - seg2.start)
                if min_dur <= total_dur <= max_dur:
                    score = (sentiment_score(seg1.text) + sentiment_score(seg2.text))
                    score += question_score(seg1.text) * 2.0
                    score -= gap * 0.1 # Penalize larger gaps
                    candidates.append({
                        "score": score,
                        "type": "multi_segment",
                        "segments": [
                            {"start": seg1.start, "end": seg1.end, "text": seg1.text.strip(), "role": "hook"},
                            {"start": seg2.start, "end": seg2.end, "text": seg2.text.strip(), "role": "payoff"}
                        ],
                        "total_duration": total_dur,
                        "gap": gap
                    })
    
    if not candidates:
        return []

    # Sort by heuristic score and return the top N
    candidates.sort(key=lambda x: x['score'], reverse=True)
    print(f"  ✓ Generated {len(candidates)} raw candidates ({len([c for c in candidates if c['type'] == 'single_extended'])} single, {len([c for c in candidates if c['type'] == 'multi_segment'])} multi). Taking top {max_candidates}.")
    return candidates[:max_candidates]

def build_scoring_prompt(candidates, video_title):
    """Builds a prompt for the AI to score pre-generated candidates."""
    
    prompt_header = f"""
You are a viral video expert. Your task is to analyze {len(candidates)} candidate clips and select the one MOST LIKELY to go viral on YouTube Shorts.

**Context:**
- Original Video Title: "{video_title}"

**Scoring Criteria:**
1.  **Hook Quality:** Does it start with a strong question, bold claim, or intriguing statement?
2.  **Narrative Flow:** Is the story satisfying? For multi-segment clips, is the payoff a logical continuation of the hook? For single-segment clips, does the "playout time" add valuable context or visual payoff?
3.  **Overall Impact:** Which clip feels the most complete, engaging, and shareable?

**Candidate Types:**
- **Single Extended:** A single piece of narration followed by silent "play out" time. Good for showcasing action.
- **Multi-Segment:** Two distinct narration segments cut together to form a story. Good for fast-paced Q&A.

**Candidate Clips:**
"""

    candidates_text = ""
    for i, cand in enumerate(candidates):
        candidates_text += f"\n--- Candidate {i+1} (Type: {cand['type']}) ---\n"
        if cand['type'] == 'single_extended':
            candidates_text += f"Narration: \"{cand['segments'][0]['text']}\"\n"
            candidates_text += f"Playout Time: {cand.get('playout_duration', 0):.1f}s of action/scenery after narration.\n"
        else: # multi_segment
            candidates_text += f"Hook: \"{cand['segments'][0]['text']}\"\n"
            candidates_text += f"Payoff: \"{cand['segments'][1]['text']}\"\n"
            candidates_text += f"Gap: {cand.get('gap', 0):.1f}s between segments.\n"

    prompt_footer = """
**Your Task:**
Respond in JSON format only. For each candidate, provide a `viral_score` from 0.0 to 10.0 and a brief `reason`. Then, identify the `best_candidate_index` (1-based).

**OUTPUT FORMAT (JSON ONLY):**
```json
{
  "scores": [
    {
      "candidate_index": 1,
      "viral_score": 8.5,
      "reason": "Excellent hook that poses a direct question, with a clear and satisfying answer."
    },
    {
      "candidate_index": 2,
      "viral_score": 6.0,
      "reason": "The hook is interesting, but the payoff is a bit weak."
    }
  ],
  "best_candidate_index": 1
}
```
"""
    return prompt_header + candidates_text + prompt_footer

# ====================== VALIDATION & LOGGING ================
def validate_segment_boundaries(clip_segments, all_words: List[Dict[str, Any]], search_window_secs=2.0) -> List[Dict[str, Any]]:
    """
    Refines clip boundaries to align with word-level timestamps for clean cuts.
    - Finds the nearest capitalized word to start a segment.
    - Finds the nearest word ending with punctuation to end a segment.
    """
    validated = []
    for clip in clip_segments:
        t0, t1 = clip["start"], clip["end"]

        # --- Find Best Start Time (t0) ---
        # Find words within a search window around the original start time
        start_candidates = [w for w in all_words if abs(w['start'] - t0) < search_window_secs]
        if start_candidates:
            # Score candidates: prefer capitalized words and proximity to original t0
            start_scores = []
            for w in start_candidates:
                score = 0
                if w['text'].strip() and w['text'].strip()[0].isupper():
                    score += 10  # Strong preference for capitalized words
                score -= abs(w['start'] - t0) # Penalize distance
                start_scores.append((score, w['start']))

            if start_scores:
                t0 = max(start_scores, key=lambda x: x[0])[1]

        # --- Find Best End Time (t1) ---
        # Find words within a search window around the original end time
        end_candidates = [w for w in all_words if abs(w['end'] - t1) < search_window_secs]
        if end_candidates:
            # Score candidates: prefer words with punctuation and proximity to original t1
            end_scores = []
            for w in end_candidates:
                score = 0
                if w['text'].strip() and w['text'].strip()[-1] in ".!?":
                    score += 10 # Strong preference for punctuation
                score -= abs(w['end'] - t1) # Penalize distance
                end_scores.append((score, w['end']))

            if end_scores:
                t1 = max(end_scores, key=lambda x: x[0])[1]

        # Ensure start is before end
        if t1 <= t0:
            print(f"  Boundary validation resulted in invalid range ({t0:.2f}s-{t1:.2f}s). Reverting to original.")
            t0, t1 = clip["start"], clip["end"]

        validated.append({"start": t0, "end": t1, "role": clip.get("role", "clip"), "text": clip.get("text", "")})
    return validated

def validate_has_words(segments: List[Dict[str, Any]], all_words: List[Dict[str, Any]]) -> bool:
    """Ensures every selected segment contains at least one transcribed word."""
    if not segments:
        print("  Validation failed: No segments provided.")
        return False
    for seg in segments:
        found_word = False
        # A word overlaps if the word's start is before the segment's end, AND the word's end is after the segment's start.
        for word in all_words:
            if seg["start"] < word["end"] and word["start"] < seg["end"]:
                found_word = True
                break
        if not found_word:
            print(f"  Validation FAILED: Segment {seg['start']:.2f}s-{seg['end']:.2f}s has no overlapping words.")
            return False
    print("  Validation PASSED: All segments have words.")
    return True


POWER_WORDS_VALIDATION = set(["secret", "truth", "never", "always", "finally", "shocking", "mistake", "fix", "proof", "hack", "trick"])
ANSWER_WORDS = set(["because", "so", "then", "that's why", "which means", "as a result"])
SOLUTION_WORDS = set(["fix", "solve", "solution", "answer", "way", "method", "system"])

def validate_micro_story(selected):
    if not selected or len(selected) not in [2, 3]:
        return False, "not_enough_segments"

    roles = [s.get("role") for s in selected]
    if len(roles) == 2 and roles != ["hook", "payoff"]:
        return False, f"invalid_role_order_2_segments: got {roles}"
    if len(roles) == 3 and roles != ["hook", "buildup", "payoff"]:
        return False, f"invalid_role_order_3_segments: got {roles}"

    # Stricter validation to match prompt's "MUST NOT exceed 40 seconds" rule.
    for a, b in zip(selected, selected[1:]):
        gap = b["start"] - a["end"]
        if gap > 65.0: # Use a slightly larger validation gap to account for model creativity
            return False, f"gap_too_large_{gap:.1f}s"

    return True, "ok_ai_validated"

def log_selection(variant: str, selected: List[Dict[str,Any]], source="ai", extra=None):
    rec = {
        "source": source, "variant": variant,
        "segments": [{"start": s["start"], "end": s["end"], "role": s.get("role","clip")} for s in selected],
        "ts": time.time(), "reason": extra or ""
    }
    try:
        with open(OUT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        print("  Log write failed:", e)

# ====================== CLIP SELECTION ======================
print("\n" + "="*20 + " CLIP SELECTION " + "="*20)
selected_segments = None
selection_method = "none"
selection_reason = "none"

channel_name, original_video_title = None, None
if URL:
    print("\n> Extracting channel/title...")
    channel_name, original_video_title = get_video_metadata(URL)
    if channel_name: print(f"  Channel: {channel_name}")
    if original_video_title: print(f"  Original Title: {original_video_title[:80]}...")

# --- AI Selection Branch (AI-ONLY) ---
if MULTI_SEG and USE_AI:
    print("\n" + "="*60)
    print(f"  USING AI MODEL: {AI_MODEL}")
    print("="*60)

    print("  ✓ Gemini library already imported.")

    api_key = os.environ.get("GOOGLE_API_KEY")
    print(f"  API key present: {bool(api_key)}")
    if not api_key:
        print("FATAL: GOOGLE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    try:
        genai.configure(api_key=api_key)
        print("  ✓ Gemini configured")
    except Exception as e:
        print(f"  ✗ Configuration failed: {e}")
        sys.exit(1)

    try:
        model = genai.GenerativeModel(AI_MODEL)
        print("  ✓ Model initialized")
    except Exception as e:
        print(f"  ✗ Model init failed: {e}")
        sys.exit(1)

    ai_segments = None
    
    # 1. Generate candidates that meet hard rules
    candidates = generate_candidates(segments_list, silences, MIN_LEN, MAX_LEN)
    if not candidates:
        print("FATAL: Could not generate any valid clip candidates. Try relaxing MIN/MAX duration or gap constraints.", file=sys.stderr)
        sys.exit(1)

    # 2. Build a prompt for the AI to score the candidates
    prompt = build_scoring_prompt(candidates, original_video_title)
    print(f"  ✓ Built scoring prompt for {len(candidates)} candidates.")

    # 3. Call the AI for scoring
    try:
        print(f"  Calling Gemini for scoring...")
        response = model.generate_content(prompt, generation_config=GenerationConfig(temperature=0.1))
        
        if not response.parts:
             print("FATAL: AI scoring returned an empty or blocked response.", file=sys.stderr)
             sys.exit(1)

        response_text = response.text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        
        scores_data = json.loads(response_text.strip())
        best_index = scores_data.get("best_candidate_index")

        print("  ✓ AI Scoring complete:")
        for score_info in scores_data.get("scores", []):
            print(f"    - Candidate {score_info['candidate_index']}: Score {score_info['viral_score']}/10 ({score_info['reason']})")

        if best_index and 1 <= best_index <= len(candidates):
            print(f"\n  AI chose Candidate #{best_index} as the winner.")
            ai_segments = candidates[best_index - 1]["segments"]
            selection_method = f"ai_only_{AI_MODEL}"
            selection_reason = "ok"
        else:
            print("FATAL: AI did not return a valid 'best_candidate_index'. Cannot proceed.", file=sys.stderr)
            log_selection("fatal_ai_choice", [], source="ai_error", extra={"reason": "invalid_best_candidate_index"})
            sys.exit(1)

    except Exception as e:
        print(f"FATAL: AI scoring call failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        log_selection("fatal_ai_choice", [], source="ai_error", extra={"reason": "api_call_failed", "error": str(e)})
        sys.exit(1)

    selected_segments = ai_segments

else:
    # This branch handles the case where USE_AI is false or MULTI_SEG is false.
    # Since we are relying exclusively on AI, we should exit if it's disabled.
    print("FATAL: This script is configured to run exclusively with the AI selector.", file=sys.stderr)
    print("       Please set USE_AI=1 and MULTI_SEGMENT=1 environment variables.", file=sys.stderr)
    sys.exit(1)

# --- Final Decision Logging & Validation ---
print("\n" + "="*20 + " FINAL DECISION & VALIDATION " + "="*20)
print(f"> Winning Method: {selection_method.upper()} ({selection_reason})")
log_selection(
    "ai_scored_multi",
    selected_segments,
    source=selection_method,
    extra=selection_reason
)

total_duration = 0.0
for i, seg in enumerate(selected_segments, 1):
    dur = seg['end'] - seg['start']; total_duration += dur
    print(f"  Segment {i} ({seg.get('role','clip')}): {seg['start']:.2f}s → {seg['end']:.2f}s ({dur:.1f}s)")

print(f"  Validating segment boundaries...")
selected_segments = validate_segment_boundaries(selected_segments, words)
print(f"  Total Duration: {total_duration:.1f}s\n")

# ====================== EXTRACT / CONCAT ===================
def extract_and_concat_segments(infile: str, segments: List[Dict[str, Any]], output_file: str):
    """Extracts multiple segments and concatenates them in a single FFmpeg command."""
    if not segments:
        raise ValueError("Segment list cannot be empty.")
    
    print(f"\n> Extracting and concatenating {len(segments)} segment(s) in one pass...")
    
    if len(segments) == 1:
        seg = segments[0]
        preset = "p6" if FFMPEG_VCODEC == "h264_nvenc" else "veryfast"
        print(f"  Single segment mode: {seg['start']:.2f}s -> {seg['end']:.2f}s")
        cmd = (f'ffmpeg -y -ss {seg["start"]:.3f} -to {seg["end"]:.3f} -i "{infile}" '
               f'-c:v {FFMPEG_VCODEC} -preset {preset} -cq 20 -c:a aac -b:a 160k -r 30 -movflags +faststart "{output_file}"')
        sh(cmd)
        return

    # Multi-segment mode
    complex_filter = []
    for i, seg in enumerate(segments):
        print(f"  Segment {i+1}: {seg['start']:.2f}s -> {seg['end']:.2f}s")
        complex_filter.append(f"[0:v]trim=start={seg['start']:.3f}:end={seg['end']:.3f},setpts=PTS-STARTPTS[v{i}];" +
                              f"[0:a]atrim=start={seg['start']:.3f}:end={seg['end']:.3f},asetpts=PTS-STARTPTS[a{i}];")
    
    concat_filter = "".join([f"[v{i}][a{i}]" for i in range(len(segments))]) + f"concat=n={len(segments)}:v=1:a=1[outv][outa]"
    preset = "p6" if FFMPEG_VCODEC == "h264_nvenc" else "veryfast"
    
    cmd = (f'ffmpeg -y -i "{infile}" -filter_complex "'
           f'{"".join(complex_filter)}{concat_filter}" '
           f'-map "[outv]" -map "[outa]" ' # Note: The FFMPEG_VCODEC is used here, which could be h264_nvenc for GPU acceleration.
           f'-c:v {FFMPEG_VCODEC} -preset {preset} -cq 20 -c:a aac -b:a 160k -r 30 -movflags +faststart "{output_file}"')
    sh(cmd)

# ====================== SUBTITLES (ASS) ====================
POWER = set(["secret","truth","never","always","finally","shocking","mistake","fix","proof"])

def stylize_word(tok: str) -> str:
    low = tok.lower().strip(".,!?")
    if SUBS_HILITE and (NUMBER_RE.match(tok) or low in POWER):
        # Apply a more dynamic animation: pop of color and size
        # \t(t1, t2, accel, tags)
        # Here, it will change to yellow and 110% size for 200ms, then back.
        effect = "\\t(0, 200, 1, {\\c&H00FFFF&\\fscx110\\fscy110})\\t(200, 400, 1, {\\c&HFFFFFF&\\fscx100\\fscy100})"
        return f"{{{effect}}}" + tok
    return tok

def build_karaoke_subtitles(words: List[Dict[str, Any]], segments: List[Dict[str, float]], ass_path: str) -> int:
    """
    Creates subtitle lines that:
    - Group 2-4 words together for readability
    - Appear ONLY when speech is happening
    - Disappear completely during silence/music/pauses
    - Work with ANY clip structure (silent intros, music, random audio)
    """
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920

    st = pysubs2.SSAStyle()
    st.name = "Karaoke"
    st.fontname = "Montserrat Black"
    st.fontsize = 85
    
    # --- FIX 1: Make Primary (un-highlighted) text TRANSPARENT ---
    # This fixes "subtitles appearing before words are spoken".
    # The pysubs2.Color(r, g, b, a) alpha '255' is fully transparent.
    st.primarycolor = pysubs2.Color(255, 255, 255, 0) # Opaque white for the base text
    
    # Secondary (highlighted) text remains opaque yellow
    st.secondarycolor = pysubs2.Color(255, 215, 0, 0) # Alpha '0' is opaque
    
    st.outlinecolor = pysubs2.Color(0, 0, 0, 0)
    st.backcolor = pysubs2.Color(0, 0, 0, 255) # Fully transparent background
    st.bold = True
    st.borderstyle = 3
    st.alignment = 2
    st.marginv = 540
    subs.styles[st.name] = st

    video_time_offset = 0.0 # Tracks the start time of the current segment in the *final* video
    total_words_processed = 0
    
    # --- FIX 2: Implement robust word-grouping and timing logic ---
    # This logic iterates through the *selected* segments, calculates
    # timestamps relative to the *new* concatenated video, and
    # naturally skips segments that have no words (i.e., music/silence).
    
    for seg_idx, seg in enumerate(segments):
        seg_start, seg_end = seg['start'], seg['end']
        seg_duration = seg_end - seg_start
        
        # Get all words that *start* within this segment's original timeframe
        seg_words = [w for w in words if seg_start <= w['start'] < seg_end]

        # If a segment has no words (e.g., it's just music or a "play out" pause),
        # we simply add its duration to the offset and add NO subtitles.
        # This solves the "I don't want subtitles during silence" problem.
        if not seg_words:
            print(f"  Segment {seg_idx+1} ({seg_start:.2f}s-{seg_end:.2f}s) has no words. Skipping (this is likely a music/pause segment).")
            video_time_offset += seg_duration
            continue

        print(f"  Segment {seg_idx+1}: Processing {len(seg_words)} words from {seg_start:.2f}s...")
        total_words_processed += len(seg_words)
        
        # Group words into lines
        i = 0
        while i < len(seg_words):
            
            # --- Start of a new line ---
            line_words = []
            
            # 1. Find words for this line (e.g., max 4 words, break on punctuation)
            for j in range(i, len(seg_words)):
                word = seg_words[j]
                line_words.append(word)
                
                # Break if line is at max length
                if len(line_words) >= 4:
                    break
                # Break if word ends a sentence (and it's not the only word)
                if len(line_words) > 1 and any(word['text'].strip().endswith(p) for p in ".!?,"):
                    break
            
            if not line_words:
                i += 1 # Should not happen, but as a safeguard
                continue
            
            line_start_time = line_words[0]['start']
            line_end_time = line_words[-1]['end']
            
            karaoke_tags = ""
            
            # 2. Build \k tags for the line
            for j, word in enumerate(line_words):
                # \k duration is the time from this word's start to the *next* word's start
                duration_s = 0
                if j < len(line_words) - 1:
                    # Duration is time until *next* word starts
                    duration_s = line_words[j+1]['start'] - word['start']
                else:
                    # Last word: duration is its own length + a small buffer
                    duration_s = (word['end'] - word['start']) + 0.20 # 200ms hang time
                
                duration_cs = max(1, int(duration_s * 100)) # min 1cs
                
                # Apply power word styling (from your stylize_word function)
                stylized_word = stylize_word(word['text'])
                
                karaoke_tags += f"{{\\k{duration_cs}}}{stylized_word}"

            # 3. Create the ASS event with adjusted timing
            event = pysubs2.SSAEvent()
            event.style = st.name
            event.text = karaoke_tags

            # --- This is the critical timing adjustment ---
            # Start time = (word_start_in_original - segment_start_in_original) + offset_in_final_video
            event.start = pysubs2.make_time(ms=(line_start_time - seg_start + video_time_offset) * 1000)
            
            # End time: Let the line persist until the *next line* starts,
            # or until the segment ends.
            
            # When does the *next* line start (in original video time)?
            next_line_start_s = float('inf')
            if i + len(line_words) < len(seg_words):
                next_line_start_s = seg_words[i + len(line_words)]['start']

            # The line should disappear when the next line starts,
            # or 2s after it finishes, whichever is *earlier*.
            end_time_s = min(line_end_time + 2.0, next_line_start_s)
            
            # But it *must* disappear by the end of the segment
            final_end_time_s = min(end_time_s, seg_end)

            # Adjust end time to the final video's timeline
            event.end = pysubs2.make_time(ms=(final_end_time_s - seg_start + video_time_offset) * 1000)
            
            # Ensure start is not after end (can happen with tiny segments)
            if event.start < event.end:
                subs.append(event)
            
            # Move to the next group of words
            i += len(line_words)
        
        # After processing all words in this segment, add its duration to the offset
        # for the *next* segment.
        video_time_offset += seg_duration

    # Save the file
    subs.save(ass_path, encoding="utf-8")
    print(f"\n✓ Karaoke .ASS file built with {total_words_processed} words.")
    print(f"  ASS path: {ass_path}")
    return total_words_processed

# ====================== RENDER =============================
if MULTI_SEG and len(selected_segments) > 1:
    temp_concat = os.path.join(OUT_DIR, "temp_concat.mp4")
    extract_and_concat_segments(INFILE, selected_segments, temp_concat)
    render_input = temp_concat
    build_karaoke_subtitles(words, selected_segments, OUT_ASS)
    input_opts = f'-i "{render_input}"'
else:
    # For single segments, we don't need a temporary concatenated file.
    # We can do the trim, scale, and subtitle burn-in all in one command.
    render_input = INFILE
    build_karaoke_subtitles(words, selected_segments, OUT_ASS)
    seg = selected_segments[0] # This is safe because the 'else' block only runs for single segments
    input_opts = f'-ss {seg["start"]:.3f} -to {seg["end"]:.3f} -i "{render_input}"'

print("\n> Applying subtitles and portrait format...")
ass_escaped = OUT_ASS.replace('\\','/').replace(':','\\:')

# Create filter complex for portrait mode with blurred background
vf_complex = (
    # Split input into two streams
    "[0:v]split=2[v1][v2];"
    # Create blurred background: scale to portrait height, then crop to width
    "[v1]scale=-1:1920,boxblur=20:10,crop=1080:1920[bg];"
    # Scale main video to fit portrait bounds
    "[v2]scale='if(gt(a,9/16),1080,-2)':'if(gt(a,9/16),-2,1920)'[fg];"
    # Overlay main video on blurred background
    "[bg][fg]overlay=(W-w)/2:(H-h)/2[v_out];"
    # Apply subtitles to the composited video
    f"[v_out]subtitles={ass_escaped}[v_final]"
)

cmd = (f'ffmpeg -y {input_opts} '
       f'-filter_complex "{vf_complex}" '
       f'-map "[v_final]" '  # Map the final video output
       f'-map 0:a? '         # Map audio if it exists
       f'-c:v {FFMPEG_VCODEC} -preset veryfast -crf 20 '
       f'-c:a aac -b:a 160k -r 30 '
       f'-movflags +faststart "{OUT_MP4}"')

print("\n> Final Render Command:")
try:
    sh(cmd)
except subprocess.CalledProcessError as e:
    print(f"\n✗ FFmpeg rendering failed!")
    print(f"  Command: {cmd}")
    print(f"  Error code: {e.returncode}")
    # Check if temp_concat exists and has valid dimensions
    probe_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "{render_input}"'
    try:
        result = subprocess.check_output(probe_cmd, shell=True, text=True)
        print(f"  Input dimensions for render: {result.strip()}")
    except:
        print(f"  Could not probe input file '{render_input}' for dimensions.")
    sys.exit(1)

print("OK ->", OUT_MP4)

# ====================== TITLES & METADATA ==================
def generate_titles_with_gemini(model, clip_text: str, original_title: Optional[str] = None, channel_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Calls the Gemini model to generate a set of viral titles and a description.
    """
    print("\n> Generating title and description with AI...")
    prompt = f"""
You are a world-class YouTube Shorts title writer, an expert in creating clickable, viral, and engaging titles.
Your task is to generate 5-10 compelling titles for a YouTube Short based on its transcript and original video title.

**== Context ==**
- **Original Video Title:** "{original_title or 'N/A'}"
- **Channel:** "{channel_name or 'N/A'}"
- **Clip Transcript:**
---
{clip_text[:1200]}...
---

**== Rules for Title Generation (Follow Strictly) ==**
1.  **Quantity:** Generate a list of 5 to 10 unique titles.
2.  **Length:** Titles MUST be between 40 and 70 characters. This is crucial for visibility on the Shorts shelf.
3.  **Hook-Driven:** Use strong hooks. Start with questions, bold statements, or create a curiosity gap.
4.  **Keywords:** Naturally include keywords from the transcript and original title.
5.  **Clarity & Punch:** Titles should be easy to understand and pack a punch. Avoid jargon.
6.  **Format:** Do NOT use clickbait tactics like "You Won't Believe...". Instead, be specific and intriguing. Use title case (e.g., "This Is a Great Title").

**== Examples of A+ Titles ==**
- The Secret to Perfect Coffee Is Not the Beans
- Why I Traded My MacBook for a Chromebook
- This 30-Second Habit Changed My Life
- The #1 Mistake Programmers Make
- How to Learn Any Skill 10x Faster

**== Your Task ==**
1.  Analyze the provided context.
2.  Brainstorm and write 5-10 titles that follow all the rules.
3.  Choose the BEST title from your list.
4.  Write a short, engaging description (2-3 sentences) for the video.
5.  Generate 5-7 relevant hashtags.

**== OUTPUT FORMAT (Strict JSON ONLY) ==**
Return **ONLY** a single, valid JSON object. Do not add any commentary before or after the JSON.

```json
{{
  "best_title": "<The single best title you generated>",
  "all_titles": [
    "<Title 1>",
    "<Title 2>",
    "<Title 3>",
    "<Title 4>",
    "<Title 5>"
  ],
  "description": "<A short, engaging description for the video.>",
  "hashtags": [
    "#Hashtag1",
    "#Hashtag2",
    "#Hashtag3"
  ]
}}
```
"""
    try:
        print(f"  Building title prompt...")
        print(f"  ✓ Title prompt built ({len(prompt)} chars)")

        print(f"  Calling Gemini for title...")
        response = model.generate_content(prompt, generation_config=GenerationConfig(temperature=0))
        print(f"  ✓ Got title response")
        response_text = response.text.strip()
        print(f"  Title response length: {len(response_text)} chars")
        print(f"  Title response preview: {response_text[:300]}")

        print(f"  ✓ Got Gemini response ({len(response_text)} chars)")
        print(f"  Response preview: {response_text[:500]}")

        # Extract JSON
        json_match = re.search(r"```(json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No JSON code block found in the response.", response_text, 0)

        parsed_json = json.loads(json_match.group(2))
        # Basic validation
        if "best_title" in parsed_json and "all_titles" in parsed_json:
            print(f"  AI Title Generation SUCCESS: Found '{parsed_json['best_title']}'")
            return parsed_json
        else:
            print("  AI title generation failed: JSON missing required keys.")
            return None
    except json.JSONDecodeError as e:
        print(f"  AI title generation failed: Invalid JSON response. Error: {e}")
        return None
    except Exception as e:
        print(f"  AI title generation failed during processing: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if selected_segments:
    # Generate title based on the content of the selected clip, not the whole video
    clip_text = ""
    for seg_bounds in selected_segments:
        # Find original transcript segments that overlap with our selected clip segments
        overlapping_segments = [s.text for s in segments_list if s.start < seg_bounds['end'] and s.end > seg_bounds['start']]
        if overlapping_segments:
            clip_text += " ".join(overlapping_segments) + " "
    
    # Use the primary AI model for title generation
    meta = generate_titles_with_gemini(model, clip_text.strip(), original_video_title, channel_name)
    if not meta:
        print("  FATAL: AI title generation failed. Cannot proceed.", file=sys.stderr)
        log_selection("fatal_title", selected_segments, source="ai_error", extra={"reason": "title_generation_failed", "model": AI_MODEL})
        sys.exit(1)

    title = meta.get("best_title")
    description = meta.get("description")
    all_titles = meta.get("all_titles", [])

    print("  Title candidates:")
    for i, v in enumerate(all_titles[:TITLE_TOP_K], 1):
        print(f"    {i}. {v}")

    with open(OUT_TITLE, "w", encoding="utf-8") as f: f.write(title)
    with open(OUT_DESC, "w", encoding="utf-8") as f: f.write(description)
    print(f"OK -> {OUT_TITLE}")
    print(f"OK -> {OUT_DESC}")

else:
    print("  FATAL: No clip segments found to generate title.", file=sys.stderr)
    sys.exit(1)

print("\nAll done ✅")