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
    "question": {
        "label": "VOICEVOX:WhiteCUL",
        "id": 23,
        "speed_scale": 1.12,
        "intonation_scale": 1.0,
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
        "title": "ガウス分布の形",
        "lines": [
            ("main", "二章三節では、連続値を表す一番重要な分布の一つ、ガウス分布を見ます。"),
            ("main", "一変数なら、形を決めるつまみは平均ミューと分散シグマ二乗です。"),
            ("main", "平均は山の中心を動かし、分散は山の広がりを変えます。"),
            ("main", "式は少し重く見えますが、指数の中では、中心からどれくらい離れたかを二乗して測っています。"),
            ("summary", "ガウス分布は、中心と広がりで不確かさを表す分布です。"),
        ],
    },
    {
        "id": "scene02",
        "title": "中心極限定理",
        "lines": [
            ("main", "ガウス分布が何度も現れる理由の一つが、中心極限定理です。"),
            ("main", "たとえば、一様分布から取った数字を平均してみます。"),
            ("main", "一個だけなら平らな分布です。二個の平均では中央が少し高くなります。"),
            ("main", "十個の平均にすると、形はかなりベル型に近づきます。"),
            ("summary", "多くの小さな偶然を足し合わせると、分布はガウス型に近づきます。"),
        ],
    },
    {
        "id": "scene03",
        "title": "多変量ガウスの幾何",
        "lines": [
            ("main", "多変量ガウス分布では、平均はベクトル、広がりは共分散行列になります。"),
            ("main", "二次元で見ると、同じ密度の点は楕円の上に並びます。"),
            ("main", "楕円の向きは共分散行列の固有ベクトルで決まり、軸の長さは固有値で決まります。"),
            ("main", "このとき中心からの距離は、普通の距離ではなく、共分散で補正したマハラノビス距離です。"),
            ("summary", "共分散行列は、広がりだけでなく、変数どうしの相関も表します。"),
        ],
    },
    {
        "id": "scene04",
        "title": "共分散の制限と限界",
        "lines": [
            ("main", "一般の共分散行列は表現力がありますが、次元が大きいとパラメータが急に増えます。"),
            ("main", "そこで、対角共分散や等方共分散のように形を制限することがあります。"),
            ("main", "対角共分散では軸に沿った楕円、等方共分散では円になります。"),
            ("main", "ただし制限しすぎると、斜めの相関を表せません。"),
            ("summary", "ガウス分布は便利ですが、単峰であることと、共分散の扱いが限界になります。"),
        ],
    },
    {
        "id": "scene05",
        "title": "条件付き分布と周辺分布",
        "lines": [
            ("main", "ガウス分布には、とても扱いやすい性質があります。"),
            ("main", "一部の変数を固定した条件付き分布も、またガウス分布になります。"),
            ("main", "一部の変数を積分して消した周辺分布も、またガウス分布です。"),
            ("main", "条件付きの平均は、固定した値に応じて線形に動きます。"),
            ("summary", "ガウス分布は、条件付けても、周辺化しても、ガウスのままです。"),
        ],
    },
    {
        "id": "scene06",
        "title": "最尤推定",
        "lines": [
            ("main", "データからガウス分布の平均と共分散を決めるには、最尤推定が使えます。"),
            ("main", "平均の最尤解は、観測データの普通の平均です。"),
            ("main", "共分散の最尤解は、平均からのズレを外積にして平均したものです。"),
            ("main", "ただし有限個のデータでは、共分散を少し小さめに見積もる性質があります。"),
            ("summary", "最尤推定は分かりやすい点推定ですが、不確かさそのものは一つの値に潰してしまいます。"),
        ],
    },
    {
        "id": "scene07",
        "title": "平均のベイズ推定",
        "lines": [
            ("main", "ベイズ推定では、平均ミューそのものにも分布を置きます。"),
            ("main", "分散が既知で、平均だけが未知なら、ガウス事前分布は共役事前分布になります。"),
            ("main", "データが増えるたびに、平均に対する事後分布は狭くなり、データの平均に近づきます。"),
            ("main", "無限に多くのデータがあれば、ベイズ推定の平均は最尤推定と同じ場所へ近づきます。"),
            ("summary", "ベイズ推定では、推定値だけでなく、推定の不確かさも残しておけます。"),
        ],
    },
    {
        "id": "scene08",
        "title": "t分布と混合ガウスへの橋渡し",
        "lines": [
            ("main", "最後に、ガウス分布の限界を二つ見ます。"),
            ("main", "外れ値があると、ガウス分布の最尤推定は大きく引っ張られます。"),
            ("main", "スチューデントのティー分布は裾が厚いので、外れ値の影響を受けにくくなります。"),
            ("main", "また、一つの山しか持てない問題には、複数のガウス分布を混ぜる混合ガウスが使えます。"),
            ("summary", "二章三節の後半は、ガウスを土台にして、より柔軟な分布へ広げていく話です。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 2.3 narration WAV files with VOICEVOX.")
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
