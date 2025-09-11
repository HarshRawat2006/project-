import os
import re
import sys
import json
import time
import webbrowser
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Callable, Optional

import requests
from dotenv import load_dotenv
from transformers import pipeline

try:
    from pytube import Search as YTSearch
except Exception:
    YTSearch = None

try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
except Exception:
    spotipy = None
    SpotifyOAuth = None

load_dotenv()

EMOTION_MODEL = os.getenv("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
CUSTOM_AUTOMATION_WEBHOOK = os.getenv("CUSTOM_AUTOMATION_WEBHOOK", "")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
SPOTIFY_OAUTH_TOKEN = os.getenv("SPOTIFY_OAUTH_TOKEN", "")
SPOTIFY_DEVICE_ID = os.getenv("SPOTIFY_DEVICE_ID", "")

@dataclass
class Analysis:
    emotion: str
    emotion_score: float
    sentiment: str
    sentiment_score: float

class EmotionAutomation:
    def __init__(self) -> None:
        self.emotion_pipeline = pipeline("text-classification", model=EMOTION_MODEL, top_k=None)
        self.sentiment_pipeline = pipeline("text-classification", model=SENTIMENT_MODEL, top_k=None)
        self.intent_patterns: Dict[str, re.Pattern] = {
            "spotify_play": re.compile(r"\b(?:open\s+)?spotify\b.*?\bplay\b\s*([\"“”']?)(.+?)\1\b", re.I),
            "youtube_play": re.compile(r"\b(?:play|open)\b.*?\b(?:on\s+)?youtube\b(?:\s*([\"“”'])(.+?)\1|\s+(.+))?$", re.I),
            "youtube_quick": re.compile(r"\b(?:play|youtube)\b\s*([\"“”']?)(.+?)\1\b", re.I),
            "google_first": re.compile(r"\b(?:google|search\s+google(?:\s+for)?)\b(?:\s*([\"“”'])(.+?)\1|\s+(.+))?$", re.I),
            "open_app": re.compile(r"\b(open|launch)\s+(notepad|calculator|vscode|spotify|word|youtube|chrome|edge|cmd|command\s+window|powershell)\b", re.I),
            "open_website": re.compile(r"\b(open|launch)\s+((?:https?://)?[\w.-]+\.[a-z]{2,}\S*)\b", re.I),
            "open_path": re.compile(r"\b(open|launch)\s+([a-zA-Z]:\\[^:<>\"|?*\n]+)\b", re.I),
            "open_special": re.compile(r"\b(open|launch)\s+(my\s*pc|this\s*pc|videos?|photos?|pictures?|documents?|downloads?|desktop|music)\b", re.I),
            "create_note": re.compile(r"\b(note|remember|jot|write down|save this)\b", re.I),
            "maps": re.compile(r"\b(map|maps|direction|navigate)\b(.*)$", re.I),
            "wiki": re.compile(r"\b(wiki|wikipedia)\b(.*)$", re.I),
            "define": re.compile(r"\b(define|meaning of)\b(.*)$", re.I),
            "email": re.compile(r"\bemail\b(.*)$", re.I),
            "search_web": re.compile(r"\b(search|look up|find)\b", re.I),
            "send_slack": re.compile(r"\b(slack|notify team|tell the team)\b", re.I),
            "run_webhook": re.compile(r"\b(trigger|webhook|zapier|ifttt|make scenario)\b", re.I),
            "system_command": re.compile(r"\bsystem\b(.*)$", re.I),
        }
        self.automation_handlers: Dict[str, Callable[[str, Analysis], Dict[str, Any]]] = {
            "spotify_play": self._handle_spotify_play,
            "youtube_play": self._handle_youtube_play,
            "youtube_quick": self._handle_youtube_quick,
            "google_first": self._handle_google_first,
            "open_app": self._handle_open_app,
            "open_website": self._handle_open_website,
            "open_path": self._handle_open_path,
            "open_special": self._handle_open_special,
            "create_note": self._handle_create_note,
            "maps": self._handle_maps,
            "wiki": self._handle_wiki,
            "define": self._handle_define,
            "email": self._handle_email,
            "search_web": self._handle_search_web,
            "send_slack": self._handle_send_slack,
            "run_webhook": self._handle_run_webhook,
            "system_command": self._handle_system_command,
        }
        self._sp: Optional["spotipy.Spotify"] = None
        if SPOTIFY_OAUTH_TOKEN and spotipy:
            self._sp = spotipy.Spotify(auth=SPOTIFY_OAUTH_TOKEN)
        elif spotipy and SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
            try:
                auth_mgr = SpotifyOAuth(
                    client_id=SPOTIFY_CLIENT_ID,
                    client_secret=SPOTIFY_CLIENT_SECRET,
                    redirect_uri=SPOTIFY_REDIRECT_URI,
                    scope="user-modify-playback-state user-read-playback-state"
                )
                self._sp = spotipy.Spotify(auth_manager=auth_mgr)
            except Exception:
                self._sp = None

    def analyze(self, text: str) -> Analysis:
        emotion = self._predict_emotion(text)
        sentiment = self._predict_sentiment(text)
        return Analysis(emotion=emotion[0], emotion_score=emotion[1], sentiment=sentiment[0], sentiment_score=sentiment[1])

    def decide_intent(self, text: str) -> Optional[str]:
        for intent, pattern in self.intent_patterns.items():
            if pattern.search(text):
                return intent
        return None

    def craft_reply(self, text: str, analysis: Analysis) -> str:
        tone = analysis.emotion
        sentiment = analysis.sentiment
        openers = {
            "anger": "I hear your frustration.",
            "disgust": "I understand that felt unpleasant.",
            "fear": "I get that this feels worrying.",
            "joy": "That’s great to hear!",
            "neutral": "Alright.",
            "sadness": "I’m sorry you’re dealing with this.",
            "surprise": "That’s unexpected!",
        }
        opener = openers.get(tone, "Understood.")
        strategy = "I’ll handle it now." if tone in ["anger", "sadness", "fear"] else ("Let’s keep that momentum." if tone == "joy" else "Here’s what I’ll do.")
        close = "All set—want anything else?" if sentiment == "positive" else ("I’ll take care of everything for you." if sentiment == "negative" else "Ready for the next step whenever you are.")
        return f"{opener} {strategy} {close}"

    def run(self, text: str, execute_actions: bool = True) -> Dict[str, Any]:
        analysis = self.analyze(text)
        intent = self.decide_intent(text)
        reply = self.craft_reply(text, analysis)
        action_result: Dict[str, Any] = {"status": "skipped", "reason": "no intent recognized"}
        if intent and intent in self.automation_handlers:
            if execute_actions:
                try:
                    action_result = self.automation_handlers[intent](text, analysis)
                except Exception as exc:
                    action_result = {"status": "error", "error": str(exc)}
            else:
                action_result = {"status": "skipped", "reason": "actions disabled", "intent": intent}
        return {"reply": reply, "analysis": analysis.__dict__, "intent": intent, "action_result": action_result}

    def _predict_emotion(self, text: str) -> Tuple[str, float]:
        preds = self.emotion_pipeline(text)[0]
        best = max(preds, key=lambda x: x["score"])
        return best["label"].lower(), float(best["score"])

    def _predict_sentiment(self, text: str) -> Tuple[str, float]:
        preds = self.sentiment_pipeline(text)[0]
        best = max(preds, key=lambda x: x["score"])
        label = str(best.get("label", "")).lower()
        if "pos" in label or "positive" in label:
            mapped = "positive"
        elif "neg" in label or "negative" in label:
            mapped = "negative"
        elif "neu" in label or "neutral" in label:
            mapped = "neutral"
        else:
            mapped = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}.get(label, "neutral")
        return mapped, float(best["score"])

    @staticmethod
    def _q(s: str) -> str:
        return s.strip().strip('"').strip("“”").strip("'")

    def _handle_spotify_play(self, text: str, _: Analysis) -> Dict[str, Any]:
        m = self.intent_patterns["spotify_play"].search(text)
        song = self._q(m.group(2) if m else "")
        if not song:
            return {"status": "error", "reason": "no song provided"}
        if self._sp:
            try:
                res = self._sp.search(q=song, type="track", limit=1)
                items = res.get("tracks", {}).get("items", [])
                if items:
                    uri = items[0]["uri"]
                    self._sp.start_playback(device_id=SPOTIFY_DEVICE_ID or None, uris=[uri])
                    return {"status": "ok", "service": "spotify", "autoplay": True, "track_uri": uri}
            except Exception:
                pass
        try_urls = [f"spotify:search:{song}", f"https://open.spotify.com/search/{requests.utils.quote(song)}"]
        for url in try_urls:
            try:
                webbrowser.open(url)
                return {"status": "ok", "service": "spotify", "autoplay": False, "url": url}
            except Exception:
                continue
        return {"status": "error", "reason": "could not open spotify"}

    def _yt_first_watch_url(self, query: str) -> Optional[str]:
        if not YTSearch:
            return None
        try:
            s = YTSearch(query)
            results = s.results or []
            if not results:
                s.get_next_results()
                results = s.results or []
            if results:
                vid = results[0].video_id
                if vid:
                    return f"https://www.youtube.com/watch?v={vid}&autoplay=1"
        except Exception:
            return None
        return None

    def _handle_youtube_play(self, text: str, _: Analysis) -> Dict[str, Any]:
        m = self.intent_patterns["youtube_play"].search(text)
        q = (m.group(2) or m.group(3) or "").strip() if m else ""
        q = self._q(q)
        if not q:
            url = "https://www.youtube.com"
            webbrowser.open(url)
            return {"status": "ok", "url": url}
        url = self._yt_first_watch_url(q)
        if url:
            webbrowser.open(url)
            return {"status": "ok", "query": q, "url": url, "autoplay": True}
        url = f"https://www.youtube.com/results?search_query={requests.utils.quote(q)}"
        webbrowser.open(url)
        return {"status": "ok", "query": q, "url": url, "autoplay": False}

    def _handle_youtube_quick(self, text: str, _: Analysis) -> Dict[str, Any]:
        m = self.intent_patterns["youtube_quick"].search(text)
        q = self._q(m.group(2) if m else "")
        if not q:
            url = "https://www.youtube.com"
            webbrowser.open(url)
            return {"status": "ok", "url": url}
        url = self._yt_first_watch_url(q)
        if url:
            webbrowser.open(url)
            return {"status": "ok", "query": q, "url": url, "autoplay": True}
        url = f"https://www.youtube.com/results?search_query={requests.utils.quote(q)}"
        webbrowser.open(url)
        return {"status": "ok", "query": q, "url": url, "autoplay": False}

    def _handle_google_first(self, text: str, _: Analysis) -> Dict[str, Any]:
        m = self.intent_patterns["google_first"].search(text)
        q = (m.group(2) or m.group(3) or "").strip() if m else ""
        q = self._q(q) or text
        if DDGS:
            try:
                with DDGS() as ddgs:
                    res = next(ddgs.text(q, max_results=1), None)
                if res and res.get("href"):
                    webbrowser.open(res["href"])
                    return {"status": "ok", "query": q, "url": res["href"], "first_result": True}
            except Exception:
                pass
        url = f"https://www.google.com/search?q={requests.utils.quote(q)}"
        webbrowser.open(url)
        return {"status": "ok", "query": q, "url": url, "first_result": False}

    def _handle_open_app(self, text: str, _: Analysis) -> Dict[str, Any]:
        m = self.intent_patterns["open_app"].search(text)
        app = (m.group(2).lower() if m else "notepad").strip()
        if app in {"cmd", "command window"}:
            subprocess.Popen("start cmd", shell=True)
            return {"status": "ok", "launched": "cmd"}
        if app == "powershell":
            subprocess.Popen("start powershell", shell=True)
            return {"status": "ok", "launched": "powershell"}
        if app == "youtube":
            url = "https://www.youtube.com"
            webbrowser.open(url)
            return {"status": "ok", "launched": "youtube", "url": url}
        if app == "chrome":
            candidates = [
                r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
                r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
                r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
            ]
            for c in candidates:
                p = os.path.expandvars(c)
                if os.path.isfile(p):
                    subprocess.Popen(f'"{p}"', shell=True)
                    return {"status": "ok", "launched": "chrome", "path": p}
            subprocess.Popen('start "" chrome', shell=True)
            return {"status": "ok", "launched": "chrome", "method": "start chrome"}
        if app == "edge":
            try:
                subprocess.Popen('start "" microsoft-edge:', shell=True)
                return {"status": "ok", "launched": "edge", "method": "protocol"}
            except Exception:
                pass
            candidates = [
                r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
                r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
            ]
            for c in candidates:
                p = os.path.expandvars(c)
                if os.path.isfile(p):
                    subprocess.Popen(f'"{p}"', shell=True)
                    return {"status": "ok", "launched": "edge", "path": p}
            return {"status": "error", "launched": "edge", "reason": "not found"}
        if app == "spotify":
            candidates = [
                r"%APPDATA%\Spotify\Spotify.exe",
                r"%LOCALAPPDATA%\Microsoft\WindowsApps\Spotify.exe",
            ]
            for c in candidates:
                p = os.path.expandvars(c)
                if os.path.isfile(p):
                    subprocess.Popen(f'"{p}"', shell=True)
                    return {"status": "ok", "launched": "spotify", "path": p}
            try:
                subprocess.Popen('start "" spotify', shell=True)
                return {"status": "ok", "launched": "spotify", "method": "start spotify"}
            except Exception:
                pass
            try:
                subprocess.Popen(r'explorer.exe shell:AppsFolder\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify', shell=True)
                return {"status": "ok", "launched": "spotify", "method": "AppsFolder"}
            except Exception:
                return {"status": "error", "launched": "spotify", "reason": "not found"}
        app_map = {
            "notepad": r"notepad.exe",
            "calculator": r"calc.exe",
            "vscode": r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe",
            "word": r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
        }
        path = os.path.expandvars(app_map.get(app, "notepad.exe"))
        subprocess.Popen(path, shell=True)
        return {"status": "ok", "launched": app, "path": path}

    def _handle_open_website(self, text: str, _: Analysis) -> Dict[str, Any]:
        m = self.intent_patterns["open_website"].search(text)
        target = m.group(2) if m else ""
        if not target:
            return {"status": "error", "reason": "no website found"}
        if not re.match(r"^https?://", target, re.I):
            target = f"https://{target}"
        webbrowser.open(target)
        return {"status": "ok", "url": target}

    def _handle_open_path(self, text: str, _: Analysis) -> Dict[str, Any]:
        m = self.intent_patterns["open_path"].search(text)
        path = (m.group(2) if m else "").strip()
        if not path:
            return {"status": "error", "reason": "no path found"}
        path = os.path.expandvars(path)
        if not os.path.exists(path):
            return {"status": "error", "reason": f"path does not exist: {path}"}
        subprocess.Popen(f'explorer.exe "{path}"', shell=True)
        return {"status": "ok", "path": path}

    def _handle_open_special(self, text: str, _: Analysis) -> Dict[str, Any]:
        m = self.intent_patterns["open_special"].search(text)
        key = (m.group(2).lower().replace(" ", "") if m else "")
        home = os.path.expanduser("~")
        special = {
            "mypc": r"shell:MyComputerFolder",
            "thispc": r"shell:MyComputerFolder",
            "videos": os.path.join(home, "Videos"),
            "video": os.path.join(home, "Videos"),
            "photos": os.path.join(home, "Pictures"),
            "pictures": os.path.join(home, "Pictures"),
            "documents": os.path.join(home, "Documents"),
            "document": os.path.join(home, "Documents"),
            "downloads": os.path.join(home, "Downloads"),
            "desktop": os.path.join(home, "Desktop"),
            "music": os.path.join(home, "Music"),
        }
        target = special.get(key)
        if not target:
            return {"status": "error", "reason": "unknown special location"}
        if target.startswith("shell:"):
            subprocess.Popen(f'explorer.exe {target}', shell=True)
            return {"status": "ok", "path": target}
        if not os.path.exists(target):
            return {"status": "error", "reason": f"path does not exist: {target}"}
        subprocess.Popen(f'explorer.exe "{target}"', shell=True)
        return {"status": "ok", "path": target}

    def _handle_create_note(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        m = re.search(r"(?:note|remember|jot|write down|save this)\b(.*)$", text, re.I)
        note = m.group(1).strip() if m and m.group(1).strip() else text
        notes_dir = os.path.join(os.getcwd(), "notes")
        os.makedirs(notes_dir, exist_ok=True)
        timestamp = int(time.time())
        file_path = os.path.join(notes_dir, f"note_{timestamp}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"[emotion={analysis.emotion}, sentiment={analysis.sentiment}]\n{note}\n")
        return {"status": "ok", "path": file_path}

    def _handle_maps(self, text: str, _: Analysis) -> Dict[str, Any]:
        m = self.intent_patterns["maps"].search(text)
        q = (m.group(2) if m else "").strip()
        if not q:
            return {"status": "error", "reason": "no place provided"}
        url = f"https://www.google.com/maps/search/{requests.utils.quote(q)}"
        webbrowser.open(url)
        return {"status": "ok", "query": q, "url": url}

    def _handle_wiki(self, text: str, _: Analysis) -> Dict[str, Any]:
        m = self.intent_patterns["wiki"].search(text)
        q = (m.group(2) if m else "").strip()
        if not q:
            url = "https://en.wikipedia.org"
            webbrowser.open(url)
            return {"status": "ok", "url": url}
        url = f"https://en.wikipedia.org/w/index.php?search={requests.utils.quote(q)}"
        webbrowser.open(url)
        return {"status": "ok", "query": q, "url": url}

    def _handle_define(self, text: str, _: Analysis) -> Dict[str, Any]:
        m = self.intent_patterns["define"].search(text)
        q = (m.group(2) if m else "").strip()
        if not q:
            return {"status": "error", "reason": "no term provided"}
        url = f"https://www.dictionary.com/browse/{requests.utils.quote(q)}"
        webbrowser.open(url)
        return {"status": "ok", "term": q, "url": url}

    def _handle_email(self, text: str, _: Analysis) -> Dict[str, Any]:
        m = self.intent_patterns["email"].search(text)
        raw = (m.group(1) if m else "").strip()
        to = ""
        subject = ""
        body = ""
        to_match = re.search(r"\bto\s+(\S+)", raw, re.I)
        if to_match:
            to = to_match.group(1)
        subj_match = re.search(r"\bsubject\s+(.+?)(?=\s+body\b|$)", raw, re.I)
        if subj_match:
            subject = subj_match.group(1).strip()
        body_match = re.search(r"\bbody\s+(.+)$", raw, re.I)
        if body_match:
            body = body_match.group(1).strip()
        from requests.utils import quote
        mailto = f"mailto:{to}?subject={quote(subject)}&body={quote(body)}"
        webbrowser.open(mailto)
        return {"status": "ok", "mailto": mailto, "to": to, "subject": subject}

    def _handle_search_web(self, text: str, _: Analysis) -> Dict[str, Any]:
        query_match = re.search(r"(?:search|look up|find)\b(.*)$", text, re.I)
        query = query_match.group(1).strip() if query_match and query_match.group(1).strip() else text
        url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
        webbrowser.open(url)
        return {"status": "ok", "query": query, "url": url}

    def _handle_send_slack(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        if not SLACK_WEBHOOK_URL:
            return {"status": "skipped", "reason": "SLACK_WEBHOOK_URL not set"}
        payload = {"text": f"[{analysis.emotion}/{analysis.sentiment}] {text}"}
        r = requests.post(SLACK_WEBHOOK_URL, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=10)
        return {"status": "ok" if r.ok else "error", "code": r.status_code}

    def _handle_run_webhook(self, text: str, analysis: Analysis) -> Dict[str, Any]:
        if not CUSTOM_AUTOMATION_WEBHOOK:
            return {"status": "skipped", "reason": "CUSTOM_AUTOMATION_WEBHOOK not set"}
        payload = {"text": text, "emotion": analysis.emotion, "sentiment": analysis.sentiment}
        r = requests.post(CUSTOM_AUTOMATION_WEBHOOK, json=payload, timeout=10)
        return {"status": "ok" if r.ok else "error", "code": r.status_code}

    def _handle_system_command(self, text: str, _: Analysis) -> Dict[str, Any]:
        m = re.search(r"\bsystem\b(.*)$", text, re.I)
        raw = (m.group(1) if m else "").strip()
        command: Optional[str] = None
        executed = False
        def build_start(target: str) -> str:
            return f'start "" {target}'
        try:
            lower = raw.lower()
            if lower.startswith(("open ", "start ")):
                arg = raw.split(" ", 1)[1].strip() if " " in raw else ""
                if not arg:
                    raise ValueError("Nothing to open.")
                if re.match(r"^https?://", arg, re.I) or re.match(r"^[\w.-]+\.[a-z]{2,}(/.*)?$", arg, re.I):
                    target = arg if re.match(r"^https?://", arg, re.I) else f"https://{arg}"
                    command = build_start(target)
                elif arg in {"youtube", "yt"}:
                    command = build_start("https://www.youtube.com")
                elif arg in {"vscode", "code"}:
                    command = build_start("code")
                elif arg in {"chrome"}:
                    command = build_start("chrome")
                elif arg in {"edge"}:
                    command = 'start "" microsoft-edge:'
                elif arg in {"notepad", "calculator", "calc", "spotify", "word"}:
                    app_map = {"notepad": "notepad", "calculator": "calc", "calc": "calc", "spotify": "spotify", "word": r'"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"'}
                    command = build_start(app_map[arg])
                else:
                    arg_q = arg if (arg.startswith('"') and arg.endswith('"')) else f'"{arg}"'
                    command = build_start(arg_q)
            elif lower.startswith("search "):
                q = raw.split(" ", 1)[1].strip()
                if not q:
                    raise ValueError("Nothing to search.")
                from requests.utils import quote
                url = f"https://www.google.com/search?q={quote(q)}"
                command = build_start(url)
            elif lower.startswith("run "):
                arg = raw.split(" ", 1)[1].strip()
                if not arg:
                    raise ValueError("Nothing to run.")
                to_run = arg if (arg.startswith('"') and arg.endswith('"')) else f'"{arg}"'
                command = to_run
            else:
                from requests.utils import quote
                url = f"https://www.google.com/search?q={quote(raw)}"
                command = build_start(url)
            subprocess.Popen(command, shell=True)
            executed = True
            return {"status": "ok", "command": command, "executed": executed}
        except Exception as e:
            return {"status": "error", "command": command, "executed": executed, "error": str(e)}

def run_loop(bot: EmotionAutomation) -> None:
    print('Type commands (e.g., open spotify and play "shape of you", play "lofi" on youtube, google "python venv"). Type "stop" to exit.')
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in {"stop", "exit", "quit"}:
            print("Goodbye.")
            break
        result = bot.run(line, execute_actions=True)
        print(f"Reply: {result['reply']}")
        print(f"Emotion: {result['analysis']['emotion']} ({result['analysis']['emotion_score']:.3f})")
        print(f"Sentiment: {result['analysis']['sentiment']} ({result['analysis']['sentiment_score']:.3f})")
        print(f"Intent: {result['intent']}")
        print(f"Action: {result['action_result']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Conversational automation with human-like replies and analysis")
    parser.add_argument("--text", "-t", type=str, help='Single run input; if omitted, enters interactive loop. Say "stop" to exit.')
    parser.add_argument("--format", "-f", choices=["json", "pretty", "reply"], default="reply")
    parser.add_argument("--no-actions", action="store_true")
    args = parser.parse_args()
    bot = EmotionAutomation()
    if args.text:
        res = bot.run(args.text, execute_actions=not args.no_actions)
        if args.format == "json":
            print(json.dumps(res, indent=2, ensure_ascii=False))
        elif args.format == "pretty":
            print(f"Reply: {res['reply']}")
            print(f"Emotion: {res['analysis']['emotion']} ({res['analysis']['emotion_score']:.3f})")
            print(f"Sentiment: {res['analysis']['sentiment']} ({res['analysis']['sentiment_score']:.3f})")
            print(f"Intent: {res['intent']}")
            print(f"Action: {res['action_result']}")
        else:
            print(res["reply"])
    else:
        run_loop(bot)