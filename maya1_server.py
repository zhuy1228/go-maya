import os
import sys
import json
import base64
import wave
import struct
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ==============================
# 常量定义 (与模型训练一致)
# ==============================
CODE_START_TOKEN_ID = 128257   # Start of Speech (SOS)
CODE_END_TOKEN_ID   = 128258   # End of Speech (EOS)
CODE_TOKEN_OFFSET   = 128266   # SNAC 编码起始偏移
SNAC_MIN_ID         = 128266
SNAC_MAX_ID         = 156937   # 128266 + (7 * 4096) - 1
SNAC_TOKENS_PER_FRAME = 7
SNAC_SAMPLE_RATE    = 24000

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "runtime", "python", "models", "maya1")

# ==============================
# 检查模型是否存在
# ==============================
def check_model_files(model_dir):
    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    for f in required:
        if not os.path.isfile(os.path.join(model_dir, f)):
            return False
    has_weights = any(
        name.endswith(".safetensors")
        for name in os.listdir(model_dir)
        if name.startswith("model-")
    ) if os.path.isdir(model_dir) else False
    return has_weights

if not check_model_files(MODEL_DIR):
    print(f"错误: 模型文件不存在: {MODEL_DIR}")
    print(f"请先运行模型下载脚本:")
    print(f"  {os.path.join(BASE_DIR, 'runtime', 'python', '3.11.9', 'python.exe')} download_maya1.py")
    sys.exit(1)

# ==============================
# 加载模型与分词器
# ==============================
print("Loading Maya1 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

print("Loading Maya1 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    model_dtype = torch.bfloat16
else:
    model_dtype = torch.float32
print(f"  Device: {device}, Dtype: {model_dtype}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=model_dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model.eval().to(device)

# ==============================
# 加载 SNAC 解码器 (可选)
# ==============================
snac_model = None
try:
    from snac import SNAC
    print("Loading SNAC 24kHz decoder...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    print("SNAC decoder ready.")
except ImportError:
    print("WARNING: 'snac' package not installed. TTS will be unavailable.")
    print("  Install with: pip install snac")
except Exception as e:
    print(f"WARNING: Failed to load SNAC decoder: {e}")


# ==============================
# 工具函数
# ==============================
def build_tts_prompt(description: str, text: str) -> str:
    """构建 Maya1 TTS 格式的 prompt"""
    content = f'<description="{description}"> {text}'
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def unpack_snac_tokens(vocab_ids):
    """将 7-token SNAC 帧解包为 3 层级编码 [L1, L2, L3]"""
    if vocab_ids and vocab_ids[-1] == CODE_END_TOKEN_ID:
        vocab_ids = vocab_ids[:-1]

    frames = len(vocab_ids) // SNAC_TOKENS_PER_FRAME
    vocab_ids = vocab_ids[:frames * SNAC_TOKENS_PER_FRAME]

    if frames == 0:
        return [[], [], []]

    l1, l2, l3 = [], [], []
    for i in range(frames):
        slots = vocab_ids[i * 7:(i + 1) * 7]
        l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend([
            (slots[1] - CODE_TOKEN_OFFSET) % 4096,
            (slots[4] - CODE_TOKEN_OFFSET) % 4096,
        ])
        l3.extend([
            (slots[2] - CODE_TOKEN_OFFSET) % 4096,
            (slots[3] - CODE_TOKEN_OFFSET) % 4096,
            (slots[5] - CODE_TOKEN_OFFSET) % 4096,
            (slots[6] - CODE_TOKEN_OFFSET) % 4096,
        ])
    return [l1, l2, l3]


@torch.inference_mode()
def snac_decode_to_audio(snac_tokens):
    """将 SNAC tokens 解码为音频 numpy 数组 (float32, 24kHz)"""
    if snac_model is None:
        raise RuntimeError("SNAC decoder not available")

    levels = unpack_snac_tokens(snac_tokens)
    if not levels[0]:
        raise RuntimeError("No valid SNAC frames to decode")

    codes = [
        torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
        for level in levels
    ]
    z_q = snac_model.quantizer.from_codes(codes)
    audio = snac_model.decoder(z_q)
    return audio[0, 0].cpu().numpy()


def audio_to_wav_bytes(audio_np, sample_rate=SNAC_SAMPLE_RATE):
    """将 float32 numpy 音频转为 WAV bytes"""
    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    import io
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


# ==============================
# HTTP 服务
# ==============================
class MayaHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._set_headers()
            self.wfile.write(json.dumps({"status": "ok", "tts_available": snac_model is not None}).encode())
            return
        self.send_error(404, "Unknown endpoint")

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        try:
            data = json.loads(body.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.send_error(400, f"Invalid request body: {str(e)}")
            return

        if self.path == "/chat":
            self._handle_chat(data)
            return

        if self.path == "/tts":
            self._handle_tts(data)
            return

        self.send_error(404, "Unknown endpoint")

    def _handle_chat(self, data):
        """处理对话请求 — 文本生成 (在 SOS token 前停止)"""
        text = data.get("text", "")
        max_tokens = data.get("max_new_tokens", 200)

        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                eos_token_id=[tokenizer.eos_token_id, CODE_START_TOKEN_ID, CODE_END_TOKEN_ID],
            )
        except Exception as e:
            self.send_error(500, f"Generation failed: {str(e)}")
            return

        # 只解码新生成的 tokens，过滤掉 SNAC 音频 tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text_tokens = [t.item() for t in new_tokens if t.item() < CODE_START_TOKEN_ID]
        result = tokenizer.decode(text_tokens, skip_special_tokens=True)

        self._set_headers()
        self.wfile.write(json.dumps({"text": result}).encode())

    def _handle_tts(self, data):
        """处理 TTS 请求 — 生成 SNAC 音频"""
        if snac_model is None:
            self.send_error(501, "TTS unavailable: 'snac' package not installed. Run: pip install snac")
            return

        text = data.get("text", "")
        description = data.get("description",
            "Realistic female voice in the 30s age with american accent. "
            "Normal pitch, warm timbre, conversational pacing, neutral tone delivery at med intensity."
        )
        max_tokens = data.get("max_new_tokens", 2000)

        prompt = build_tts_prompt(description, text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=[CODE_END_TOKEN_ID],
            )
        except Exception as e:
            self.send_error(500, f"Generation failed: {str(e)}")
            return

        # 提取 SNAC tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        snac_tokens = [t.item() for t in generated if SNAC_MIN_ID <= t.item() <= SNAC_MAX_ID]

        if len(snac_tokens) < SNAC_TOKENS_PER_FRAME:
            self.send_error(500, "Model did not generate enough audio tokens")
            return

        try:
            audio_np = snac_decode_to_audio(snac_tokens)
            wav_bytes = audio_to_wav_bytes(audio_np)
            b64 = base64.b64encode(wav_bytes).decode()
        except Exception as e:
            self.send_error(500, f"Audio decoding failed: {str(e)}")
            return

        self._set_headers()
        self.wfile.write(json.dumps({
            "audio_base64": b64,
            "sample_rate": SNAC_SAMPLE_RATE,
            "duration_seconds": round(len(audio_np) / SNAC_SAMPLE_RATE, 2),
        }).encode())


def run():
    server = HTTPServer(("127.0.0.1", 5005), MayaHandler)
    print("Maya1 server running at http://127.0.0.1:5005")
    server.serve_forever()

if __name__ == "__main__":
    run()
