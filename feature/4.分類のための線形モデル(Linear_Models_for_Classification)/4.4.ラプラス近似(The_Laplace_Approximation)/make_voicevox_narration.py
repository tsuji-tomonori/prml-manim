from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import wave
from pathlib import Path


SCENE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCENE_DIR / "assets" / "voicevox"

SPEAKERS = {
    "main": {
        "label": "VOICEVOX:WhiteCUL",
        "id": 23,
        "speed_scale": 1.08,
        "intonation_scale": 0.95,
    },
    "focus": {
        "label": "VOICEVOX:WhiteCUL",
        "id": 23,
        "speed_scale": 1.06,
        "intonation_scale": 0.9,
    },
    "summary": {
        "label": "VOICEVOX:WhiteCUL",
        "id": 23,
        "speed_scale": 1.04,
        "intonation_scale": 0.9,
    },
}

PAUSE_SECONDS = 0.32

SCENES = [
    {
        "id": "scene01",
        "title": "なぜラプラス近似か",
        "lines": [
            ("main", "四章四節では、複雑な確率分布を、扱いやすいガウス分布で近似するラプラス近似を見ます。"),
            ("main", "ベイズロジスティック回帰では、パラメータの事後分布がガウスにならず、積分をきれいに解けません。"),
            ("main", "そこで分布の山、つまりモードの近くを拡大し、そこだけをガウスで置き換えます。"),
            ("focus", "見たいのは、分布全体を完全に再現することではなく、中心と曲率から局所的な不確かさを読むことです。"),
            ("summary", "ラプラス近似は、難しい事後分布を、平均と共分散を持つガウス近似へ変換する入口になります。"),
        ],
    },
    {
        "id": "scene02",
        "title": "1変数の手順",
        "lines": [
            ("main", "一変数で、正規化されていない密度を f z、未知の正規化定数を Z と書きます。"),
            ("main", "まず f z が最大になる点 z ゼロを探します。ここでは微分がゼロです。"),
            ("main", "次に、ログ f z を z ゼロの周りで二次までテイラー展開します。"),
            ("main", "一次の項は、モードでは傾きがゼロなので消えます。"),
            ("summary", "指数を取り直すと、中心 z ゼロ、精度 A のガウス分布が現れます。"),
        ],
    },
    {
        "id": "scene03",
        "title": "曲率は不確かさを決める",
        "lines": [
            ("main", "ラプラス近似は、負の対数密度を見るとさらに直感的です。"),
            ("main", "山の頂上は、負の対数で見ると谷底になります。"),
            ("main", "谷底の周りを二次関数で近似すると、その曲がり具合が A です。"),
            ("main", "A が大きいほど谷は急で、ガウス近似は細くなります。"),
            ("summary", "A が小さいほど谷は緩く、推定の不確かさは広がります。"),
        ],
    },
    {
        "id": "scene04",
        "title": "多変数では Hessian が精度行列になる",
        "lines": [
            ("main", "多変数では、z はベクトルになり、モード z ゼロの周りでログ密度を二次形式にします。"),
            ("main", "このとき現れる行列 A は、ログ f の二階微分、つまり Hessian にマイナスを付けたものです。"),
            ("main", "A はガウス分布の精度行列で、共分散行列は A の逆行列になります。"),
            ("main", "楕円が細い方向は、曲率が大きく、不確かさが小さい方向です。"),
            ("summary", "A が正定値でなければ、そこは最大点ではなく、ガウス近似としては使えません。"),
        ],
    },
    {
        "id": "scene05",
        "title": "Z とモデル証拠",
        "lines": [
            ("main", "ラプラス近似は、分布の形だけでなく、積分の近似にも使えます。"),
            ("main", "f z をモード周りのガウス形で置き換えると、積分 Z は最大値と幅の積として近似できます。"),
            ("main", "ベイズモデル比較では、この Z がモデル証拠に対応します。"),
            ("main", "ただ高く当てはまるモデルだけでなく、許されるパラメータ範囲が狭すぎないかも効いてきます。"),
            ("summary", "この幅による補正が、Occam factor と呼ばれる複雑さへの調整です。"),
        ],
    },
    {
        "id": "scene06",
        "title": "BIC への橋渡し",
        "lines": [
            ("main", "さらに粗い近似を置くと、ラプラス近似から BIC が出てきます。"),
            ("main", "対数証拠は、最適化した尤度から、パラメータ数 M に比例する項を引いた形になります。"),
            ("main", "罰則は二分の M 掛けるログ N で、N はデータ数です。"),
            ("main", "パラメータが多いほど、またデータが多いほど、偶然の当てはまりには厳しくなります。"),
            ("summary", "BIC は計算しやすい一方で、Hessian が十分な階数を持つ、という仮定には注意が必要です。"),
        ],
    },
    {
        "id": "scene07",
        "title": "何が得意で、何が苦手か",
        "lines": [
            ("main", "ラプラス近似の良い点は、真の正規化定数 Z を知らなくても使えることです。"),
            ("main", "データが多く、事後分布が一つの鋭い山に近いときには、特に有効です。"),
            ("main", "ただし多峰性や強い歪みのような、分布全体の性質は苦手です。"),
            ("main", "正の値だけを取る変数では、ログを取るなど、変数変換をしてから近似することもあります。"),
            ("summary", "次の四章五節では、この道具を使って、ベイズロジスティック回帰の事後分布を扱います。"),
        ],
    },
]


def post_json(base_url: str, path: str, params: dict[str, str | int], body: bytes | None = None) -> bytes:
    url = f"{base_url.rstrip('/')}{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, data=body, method="POST")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def synthesize(base_url: str, speaker_key: str, text: str) -> bytes:
    speaker = SPEAKERS[speaker_key]
    query_bytes = post_json(base_url, "/audio_query", {"text": text, "speaker": speaker["id"]})
    query = json.loads(query_bytes.decode("utf-8"))
    query["speedScale"] = speaker["speed_scale"]
    query["intonationScale"] = speaker["intonation_scale"]
    query["prePhonemeLength"] = 0.12
    query["postPhonemeLength"] = 0.22
    return post_json(
        base_url,
        "/synthesis",
        {"speaker": speaker["id"]},
        json.dumps(query, ensure_ascii=False).encode("utf-8"),
    )


def append_wav_bytes(output: wave.Wave_write, wav_bytes: bytes, expected_params: tuple[int, int, int] | None) -> tuple[int, int, int]:
    tmp_path = OUTPUT_DIR / "_line.wav"
    tmp_path.write_bytes(wav_bytes)
    try:
        with wave.open(str(tmp_path), "rb") as source:
            params = (source.getnchannels(), source.getsampwidth(), source.getframerate())
            if expected_params is None:
                output.setnchannels(params[0])
                output.setsampwidth(params[1])
                output.setframerate(params[2])
            elif params != expected_params:
                raise RuntimeError(f"Unexpected WAV params: {params}, expected {expected_params}")
            output.writeframes(source.readframes(source.getnframes()))
            return params
    finally:
        tmp_path.unlink(missing_ok=True)


def append_silence(output: wave.Wave_write, params: tuple[int, int, int], seconds: float) -> None:
    channels, sample_width, frame_rate = params
    frame_count = int(frame_rate * seconds)
    output.writeframes(b"\x00" * frame_count * channels * sample_width)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def generate_scene(base_url: str, scene: dict[str, object]) -> dict[str, object]:
    scene_id = str(scene["id"])
    output_path = OUTPUT_DIR / f"{scene_id}.wav"
    tmp_output_path = OUTPUT_DIR / f".{scene_id}.tmp.wav"
    params: tuple[int, int, int] | None = None
    with wave.open(str(tmp_output_path), "wb") as output:
        for line_number, (speaker_key, text) in enumerate(scene["lines"], start=1):  # type: ignore[index]
            print(f"{scene_id} line {line_number}", flush=True)
            wav_bytes = synthesize(base_url, speaker_key, text)
            params = append_wav_bytes(output, wav_bytes, params)
            append_silence(output, params, PAUSE_SECONDS)
            time.sleep(0.03)
    tmp_output_path.replace(output_path)
    return {
        "id": scene_id,
        "title": scene["title"],
        "path": str(output_path.relative_to(SCENE_DIR)),
        "duration": round(wav_duration(output_path), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PRML 4.4 narration WAV files with VOICEVOX.")
    parser.add_argument("--base-url", default="http://127.0.0.1:50021", help="VOICEVOX Engine URL")
    parser.add_argument("--from-scene", default="scene01", help="First scene id to regenerate")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_index = next((i for i, scene in enumerate(SCENES) if scene["id"] == args.from_scene), 0)
    manifest = {
        "speakers": {key: {"label": value["label"], "id": value["id"]} for key, value in SPEAKERS.items()},
        "scenes": [],
    }
    for scene in SCENES[:start_index]:
        output_path = OUTPUT_DIR / f"{scene['id']}.wav"
        result = {
            "id": scene["id"],
            "title": scene["title"],
            "path": str(output_path.relative_to(SCENE_DIR)),
            "duration": round(wav_duration(output_path), 3),
        }
        manifest["scenes"].append(result)
    for scene in SCENES[start_index:]:
        result = generate_scene(args.base_url, scene)
        manifest["scenes"].append(result)
        print(f"{result['id']} {result['duration']}s {result['path']}", flush=True)

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest {manifest_path.relative_to(SCENE_DIR)}", flush=True)


if __name__ == "__main__":
    main()
