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
        "title": "指数型分布族とは",
        "lines": [
            ("main", "二値のベルヌーイ分布。カテゴリの多項分布。連続値のガウス分布。"),
            ("main", "見た目は違いますが、ピーアールエムエルでは、これらを同じ形で扱えることを示します。"),
            ("main", "その共通の形が、指数型分布族です。"),
            ("main", "この回では、分布の見た目ではなく、式の部品に注目します。"),
            ("main", "自然パラメータ、十分統計量、正規化。三つの部品が見えれば、節全体の流れがつかめます。"),
        ],
    },
    {
        "id": "scene02",
        "title": "標準形の部品",
        "lines": [
            ("main", "指数型分布族の標準形は、この式です。"),
            ("main", "エイチエックスは、エックスだけに依存する土台です。"),
            ("main", "ユーエックスは、データから取り出す特徴量です。これを十分統計量と呼びます。"),
            ("main", "イータは、自然パラメータです。ふつうの平均や分散とは違う、式がまっすぐになる座標です。"),
            ("main", "ジーイータは、全体を確率分布にするための正規化係数です。"),
            ("summary", "指数の中では、パラメータとデータの特徴量が内積で結びついています。"),
        ],
    },
    {
        "id": "scene03",
        "title": "ベルヌーイ分布の書き換え",
        "lines": [
            ("main", "まず、ベルヌーイ分布で見てみます。"),
            ("main", "エックスはゼロか一。確率ミューで一が出る分布です。"),
            ("main", "この式を指数型分布族の形に直すと、自然パラメータは、ミュー割る一マイナスミューの対数になります。"),
            ("main", "これはロジット、つまりオッズの対数です。"),
            ("main", "ミューを動かすと、一の棒とゼロの棒の高さが入れ替わります。"),
            ("summary", "ベルヌーイでは、十分統計量はエックスそのものです。"),
        ],
    },
    {
        "id": "scene04",
        "title": "ガウス分布も同じ枠に入る",
        "lines": [
            ("main", "次に、分散を固定した一変量ガウス分布を見ます。"),
            ("main", "いつもの式には、エックスマイナス平均の二乗が入っています。"),
            ("main", "二乗を展開すると、エックスの項と、エックス二乗の項に分かれます。"),
            ("main", "つまり十分統計量は、エックスとエックス二乗の二つです。"),
            ("main", "平均を動かすと山の位置は動きますが、指数型分布族としては自然パラメータの座標で動いている、と見られます。"),
        ],
    },
    {
        "id": "scene05",
        "title": "十分統計量がデータを圧縮する",
        "lines": [
            ("main", "指数型分布族では、データ全体は十分統計量の和として効いてきます。"),
            ("main", "ベルヌーイなら、必要なのは一が何回出たかです。"),
            ("main", "ガウスなら、エックスの和と、エックス二乗の和が中心になります。"),
            ("question", "全部のデータを覚えなくてもいい、ということですか？"),
            ("main", "この分布族のパラメータ推定に関しては、はい。必要な情報が統計量に圧縮されます。"),
            ("summary", "十分統計量は、推定に必要なデータの要約です。"),
        ],
    },
    {
        "id": "scene06",
        "title": "最尤推定は平均合わせになる",
        "lines": [
            ("main", "最尤推定では、尤度が最大になるイータを探します。"),
            ("main", "指数型分布族では、その条件がきれいな形になります。"),
            ("main", "モデルが予測する十分統計量の平均を、データから計算した十分統計量の平均に合わせる。"),
            ("main", "ベルヌーイなら、モデルの平均ミューを、観測された一の割合に合わせるだけです。"),
            ("main", "複雑に見える最尤推定が、平均を合わせる操作として読めるのが大きな利点です。"),
        ],
    },
    {
        "id": "scene07",
        "title": "共役事前分布",
        "lines": [
            ("main", "ベイズ推論では、パラメータにも事前分布を置きます。"),
            ("main", "指数型分布族には、形が保たれる共役事前分布があります。"),
            ("main", "事前分布の中にも、自然パラメータと十分統計量に対応する部品が入っています。"),
            ("main", "データを観測すると、事前の要約量に、データの十分統計量の和を足すだけで更新できます。"),
            ("summary", "共役性は、ベイズ更新を足し算として扱えるようにします。"),
        ],
    },
    {
        "id": "scene08",
        "title": "2.4節のまとめ",
        "lines": [
            ("main", "指数型分布族は、一つの新しい分布ではありません。"),
            ("main", "ベルヌーイ、カテゴリ、ガウスなどを、同じ式で見るための枠組みです。"),
            ("main", "鍵は、自然パラメータ、十分統計量、正規化係数の三つです。"),
            ("main", "この形に入ると、最尤推定、共役事前分布、ベイズ更新が同じ記法で扱えます。"),
            ("main", "次の節で扱うノンパラメトリック手法とは対照的に、ここでは有限個のパラメータで分布を表します。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 2.4 narration WAV files with VOICEVOX.")
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
