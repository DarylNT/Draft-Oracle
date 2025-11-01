import os, time, json, random, datetime, logging
import requests
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

LOG_FILE = "collector.log"
os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)

logger = logging.getLogger("collector")
logger.setLevel(logging.INFO)

# rotating file handler (5 MB per file, 3 backups)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(file_handler)

# console handler for critical messages
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.CRITICAL)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(console_handler)

load_dotenv()
API_KEY   = os.getenv("RIOT_API_KEY", "YOUR_RIOT_API_KEY")
PLATFORM  = "na1"        
REGION    = "americas"   
PATCH     = "15.21"
PATCH_TS = 1761127200  # Patch 15.21 (25.21) release: 2025-10-22 03:00 AM PT (10:00 UTC)
QUEUE_ID  = 420
OUTPUT    = "matches_25_21.jsonl"
CHECKPOINT_FILE = "seen_ids.json"
CLASS_MAP = "champ_class_map.json"

SESSION = requests.Session()
SESSION.headers.update({"X-Riot-Token": API_KEY})

MIN_TIME = 1.25
MAX_TIME = 1.35

"""
    Handles proper API call timing (100 calls/120s)
    Takes in the API call function and tracks the how long between each call in order to confirm timing
"""
def throttle_handler(func, last_time, *args, **kwargs):
    if last_time is not None:
        elapsed = time.perf_counter() - last_time
        remaining = random.uniform(MIN_TIME, MAX_TIME) - elapsed
        if remaining > 0:
            logger.info(f"Sleeping {remaining:.2f}s (elapsed {elapsed:.2f}s)")
            time.sleep(remaining)

    data = func(*args, **kwargs)
    return data, time.perf_counter()

def safe_get(url, max_retries=3, timeout=8): # Back-up safety net in case API returns bad status codes
    backoff = 2
    for _ in range(max_retries):
        try:
            resp = SESSION.get(url, timeout=timeout)
        except requests.exceptions.RequestException as e:
            logger.warning(f"[REQ ERR] {e}")
            time.sleep(backoff)
            backoff *= 1.5
            continue

        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "2"))
            logger.warning(f"[429] Sleeping {retry_after}s…")
            time.sleep(retry_after + 1)
            continue
        if resp.status_code >= 500:
            logger.warning(f"[{resp.status_code}] Server error, retrying...")
            time.sleep(backoff)
            backoff *= 1.5
            continue

        logger.error(f"[{resp.status_code}] {resp.text[:100]}")
    return None

# Track Matche Ids (Anti-Duplicate Technology) ------------------------------------------

def load_seen_ids():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_seen_ids(seen):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(list(seen), f, indent=2)

# API CALS ----------------------------------------------------------

BASE_PLATFORM = f"https://{PLATFORM}.api.riotgames.com"
BASE_REGION   = f"https://{REGION}.api.riotgames.com"

def get_league_entries(tier):
    url = f"{BASE_PLATFORM}/lol/league/v4/{tier}leagues/by-queue/RANKED_SOLO_5x5"
    data = safe_get(url)
    if not data:
        logger.warning(f"No data returned for {tier}")
        return []
    logger.info(f"Fetched {len(data.get('entries', []))} entries for {tier}")
    return data.get("entries", [])

def get_match_ids(puuid, start=0, count=100):
    url = (f"{BASE_REGION}/lol/match/v5/matches/by-puuid/{puuid}/ids"
           f"?queue={QUEUE_ID}&startTime={PATCH_TS}&start={start}&count={count}")
    return safe_get(url) or []

def get_match_detail(mid):
    url = f"{BASE_REGION}/lol/match/v5/matches/{mid}"
    return safe_get(url)

# Process Match ---------------------------------------------------------

def normalize_role(pos):
    return {
        "TOP": "Top", "JUNGLE": "Jungle", "MIDDLE": "Mid",
        "MID": "Mid", "BOTTOM": "ADC", "BOT": "ADC",
        "UTILITY": "Support", "SUPPORT": "Support"
    }.get(pos.upper(), "Unknown")

def load_class_map(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def class_info(champ, cmap):
    d = cmap.get(champ)
    if d:
        return d.get("major", "Unknown"), d.get("sub", "Unknown")
    return "Unknown", "Unknown"

def transform_match(match_json, cmap):
    try:
        info = match_json.get("info", {})
        meta = match_json.get("metadata", {})

        full_version = info.get("gameVersion", "")
        version = ".".join(info.get("gameVersion", "").split(".")[:2])
        if not full_version.startswith(PATCH) or info.get("queueId") != QUEUE_ID or info.get("gameDuration", 0) < 600:
            logger.info(f"Skipping: gameVersion={full_version}")
            return None
        else:
            logger.info(f"Did not skip: gameVersion={full_version}")


        blue, red = {}, {}
        for p in info.get("participants", []):
            team = blue if p.get("teamId") == 100 else red
            role = normalize_role(p.get("individualPosition") or p.get("teamPosition") or "")
            if role == "Unknown": continue
            champ = p.get("championName", "Unknown")
            major, sub = class_info(champ, cmap)
            team[role] = {"champion": champ, "class": major, "subclass": sub}

        if not ({"Top","Jungle","Mid","ADC","Support"} <= blue.keys() and {"Top","Jungle","Mid","ADC","Support"} <= red.keys()):
            logger.debug(f"Skipping match {meta.get('matchId')} due to missing roles: blue={list(blue.keys())}, red={list(red.keys())}")
            return None

        blue_win = any(p.get("win") and p.get("teamId") == 100 for p in info.get("participants", []))

        return {
            "matchId": meta.get("matchId"),
            "patch": version,
            "gameCreation": info.get("gameCreation"),
            "region": PLATFORM,
            "queueId": info.get("queueId"),
            "duration": info.get("gameDuration"),
            "blue": blue,
            "red": red,
            "winner": "blue" if blue_win else "red"
        }
    except KeyError as e:
        logger.exception(f"Error during transform_match: {e}")
        time.sleep(2)

# Where the magic happens -------------------------------------------

def collect_matches(seen):
    cmap = load_class_map(CLASS_MAP)
    saved = 0
    checkpoint_interval = 100
    last_call_time = None

    with open(OUTPUT, "a", encoding="utf-8") as f:
        for tier in ("challenger", "grandmaster", "master"):
            logger.info(f"Fetching {tier} players…")
            time.sleep(random.uniform(2,3))
            players, last_call_time = throttle_handler(get_league_entries, last_call_time, tier)
            if not players: continue

            for pl in players:
                puuid = pl.get("puuid")
                if not puuid: 
                    logger.warning(f"No puuid found for player entry: {pl}")
                    continue

                start = 0
                while True:
                    ids, last_call_time = throttle_handler(get_match_ids, last_call_time, puuid, start=start, count=checkpoint_interval)
                    if not ids: break
                    new = 0
                    for mid in ids:
                        if mid in seen: continue
                        seen.add(mid)
                        match, last_call_time = throttle_handler(get_match_detail, last_call_time, mid)
                        if not match: continue
                        rec = transform_match(match, cmap)
                        if not rec: continue
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        saved += 1
                        new += 1
                        if saved % checkpoint_interval == 0:
                            save_seen_ids(seen)
                            logger.info(f"Checkpoint: {saved} matches saved")
                        logger.info(f"[{datetime.datetime.now():%H:%M:%S}][{saved}] Saved {mid}")

                    if new == 0: break
                    start += 100

    save_seen_ids(seen)
    logger.info(f"Done. Saved {saved} matches total -> {OUTPUT}")

# int main() {
try:
    seen = load_seen_ids()
    collect_matches(seen)
except KeyboardInterrupt:
    save_seen_ids(seen)
    logger.critical("Manual stop - Progress saved")
# return 0;
# }