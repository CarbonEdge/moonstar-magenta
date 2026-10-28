"""Live playback: generate chunks on the GPU and stream them to a browser; change the prompt while it plays.

Stdlib only. Run in the container with the port published, then open http://localhost:8765 :

    python -m moonstar_magenta.live --prompt "lo-fi hip hop" --seed 7 --cfg 6
"""
import argparse
import json
import queue
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from moonstar_magenta.stream import FRAMES_PER_SEC, embed_spec

SAMPLE_RATE, CHANNELS = 48000, 2

PAGE = """<!doctype html><meta charset=utf-8><title>moonstar-magenta live</title>
<body style="font:16px system-ui;max-width:640px;margin:2rem auto;padding:0 1rem">
<h3>moonstar-magenta live</h3>
<p><button id=go onclick="start()">Play</button> <button onclick="stop()">Stop</button> <span id=b></span></p>
<p><input id=p style="width:100%" placeholder="0.7*file:data/samples/ref1.wav | 0.3*funky bass"></p>
<p><button onclick="send()">Apply prompt</button> <span id=s></span></p>
<script>
const p=document.getElementById("p"),s=document.getElementById("s");
async function send(){await fetch("/prompt",{method:"POST",body:p.value});poll()}
async function poll(){const r=await (await fetch("/status")).json();s.textContent="now: "+r.prompt+" | lead "+r.lead.toFixed(1)+"s";}
p.addEventListener("keydown",e=>{if(e.key==="Enter")send()});
setInterval(poll,2000);poll();

// Web Audio player: read raw PCM from the stream, schedule gapless buffers, cap how far ahead we buffer.
const SR=48000,CH=2,FRAME=4,MAX_AHEAD=2.5,START_PAD=0.4;
let ctx=null,ctl=null,next=0,bsp=document.getElementById("b");
async function start(){
  stop();ctx=new AudioContext({sampleRate:SR});ctl=new AbortController();
  const my=ctx,res=await fetch("/audio.wav",{signal:ctl.signal}),rd=res.body.getReader();
  let buf=new Uint8Array(0),skip=44;next=0;
  try{while(ctx===my){
    const {value,done}=await rd.read();if(done)break;
    let v=value;if(skip){const n=Math.min(skip,v.length);v=v.subarray(n);skip-=n;}
    const m=new Uint8Array(buf.length+v.length);m.set(buf);m.set(v,buf.length);
    const n=m.length-(m.length%FRAME);buf=m.slice(n);if(!n)continue;
    const pcm=new Int16Array(m.buffer,0,n/2),frames=n/FRAME,ab=my.createBuffer(CH,frames,SR);
    for(let c=0;c<CH;c++){const d=ab.getChannelData(c);for(let i=0;i<frames;i++)d[i]=pcm[i*CH+c]/32768;}
    const src=my.createBufferSource();src.buffer=ab;src.connect(my.destination);
    if(next<my.currentTime)next=my.currentTime+START_PAD;
    src.start(next);next+=ab.duration;
    bsp.textContent="buffered "+(next-my.currentTime).toFixed(1)+"s";
    while(ctx===my&&next-my.currentTime>MAX_AHEAD)await new Promise(r=>setTimeout(r,100));
  }}catch(e){if(e.name!=="AbortError")bsp.textContent="stream error: "+e;}
}
function stop(){if(ctl)ctl.abort();if(ctx)ctx.close();ctx=null;ctl=null;bsp.textContent="";}
</script>"""


def wav_header(sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> bytes:
    """16-bit PCM header with a maximal length, so the browser treats the stream as endless."""
    block = channels * 2
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, sample_rate * block, block, 16) +
            b"data" + struct.pack("<I", 0xFFFFFFFF))


def to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def sleep_needed(produced_sec: float, elapsed_sec: float, lead_sec: float) -> float:
    """Seconds to wait so generation stays at most `lead_sec` ahead of the playhead."""
    return max(0.0, produced_sec - elapsed_sec - lead_sec)


class Session:
    """Generator thread: produces PCM chunks into a bounded queue; the prompt can change at any time."""

    def __init__(self, mrt, Waveform, MUSICCOCA, args):
        self.mrt, self.Waveform, self.key, self.args = mrt, Waveform, MUSICCOCA.key, args
        self.prompt = args.prompt
        self.pending = None
        self.chunks: queue.Queue = queue.Queue(maxsize=2)
        self.produced = 0.0
        self.t0 = None
        self.embeddings: dict[str, object] = {}

    def set_prompt(self, text: str) -> None:
        self.pending = text.strip() or None

    def status(self) -> dict:
        lead = self.produced - (time.perf_counter() - self.t0) if self.t0 else 0.0
        return {"prompt": self.prompt, "lead": lead}

    def rebase(self) -> None:
        """New listener: drop stale buffered chunks and restart pacing so playback begins near-live."""
        while True:
            try:
                self.chunks.get_nowait()
            except queue.Empty:
                break
        if self.t0 is not None:
            self.t0 = time.perf_counter() - self.produced

    def _embed(self, spec: str):
        if spec not in self.embeddings:
            self.embeddings[spec] = embed_spec(self.mrt, spec, self.Waveform, self.args.seed)
        return self.embeddings[spec]

    def run(self) -> None:
        frames = int(self.args.chunk_sec * FRAMES_PER_SEC)
        state, emb = None, self._embed(self.prompt)
        self.t0 = time.perf_counter()
        while True:
            if self.pending and self.pending != self.prompt:
                try:
                    emb, self.prompt = self._embed(self.pending), self.pending
                    print(f"[{self.produced:6.1f}s] prompt -> {self.prompt!r}", flush=True)
                except Exception as e:  # bad file path etc: keep playing the old prompt
                    print(f"prompt rejected: {e}", flush=True)
                self.pending = None
            time.sleep(sleep_needed(self.produced, time.perf_counter() - self.t0, self.args.lead))
            wav, state = self.mrt.generate(
                conditioning={self.key: emb}, frames=frames, state=state,
                temperature=self.args.temperature, top_k=self.args.top_k,
                cfg_scales={"musiccoca": self.args.cfg} if self.args.cfg else None)
            self.chunks.put(to_pcm16(wav.samples))
            self.produced += self.args.chunk_sec


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/status":
                self._send(200, "application/json", json.dumps(session.status()).encode())
            elif self.path == "/audio.wav":
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                session.rebase()
                try:
                    self.wfile.write(wav_header())
                    while True:
                        self.wfile.write(session.chunks.get())
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(200, "text/html; charset=utf-8", PAGE.encode())

        def do_POST(self):
            if self.path == "/prompt":
                session.set_prompt(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
                self._send(200, "text/plain", b"ok")
            else:
                self._send(404, "text/plain", b"")
    return Handler


def main() -> None:
    ap = argparse.ArgumentParser("live")
    ap.add_argument("--model", default="mrt2_small")
    ap.add_argument("--prompt", default="lo-fi hip hop, mellow piano")
    ap.add_argument("--chunk-sec", type=float, default=1.0)
    ap.add_argument("--lead", type=float, default=2.0, help="max seconds generated ahead of playback")
    ap.add_argument("--seed", type=int, default=0, help="mapper noise seed for text prompts")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    from magenta_rt import MagentaRT2Jax
    from magenta_rt.audio import Waveform
    from magenta_rt.config import MUSICCOCA

    session = Session(MagentaRT2Jax(size=args.model), Waveform, MUSICCOCA, args)
    threading.Thread(target=session.run, daemon=True).start()
    print(f"live on http://localhost:{args.port}  (seed={args.seed}, cfg={args.cfg})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(session)).serve_forever()


if __name__ == "__main__":
    main()
