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

PAUSE_SECONDS = 0.30

SCENES = [
    {
        "id": "scene01",
        "title": "なぜ分解するのか",
        "lines": [
            ("main", "今回は、ピーアールエムエル三点二、バイアス、バリアンス分解です。"),
            ("main", "複雑なモデルは、訓練データにはよく合いますが、データが少ないと過学習しやすくなります。"),
            ("main", "一方で、単純すぎるモデルは、どのデータで学習しても本当の曲線を追えません。"),
            ("question", "この二つの失敗を、どう切り分けて考えればよいのでしょうか。"),
            ("summary", "答えが、平均的なズレであるバイアスと、データによる揺れであるバリアンスです。"),
        ],
    },
    {
        "id": "scene02",
        "title": "損失の出発点",
        "lines": [
            ("main", "回帰で二乗損失を使うと、理想の予測は条件付き平均、エイチエックスになります。"),
            ("main", "予測ワイエックスと、エイチエックスの差が小さいほど、モデル由来の損失は小さくなります。"),
            ("main", "ただし観測値ティーには、どうしても残るノイズがあります。"),
            ("summary", "したがって期待損失は、モデルが改善できる部分と、改善できないノイズ部分に分けられます。"),
        ],
    },
    {
        "id": "scene03",
        "title": "データ集合を取り替える思考実験",
        "lines": [
            ("main", "頻度主義の見方では、同じ大きさの訓練データを何度も取り直す、と考えます。"),
            ("main", "それぞれのデータ集合で学習すると、少しずつ違う予測関数が得られます。"),
            ("main", "この予測関数たちを平均したものを、平均予測と呼ぶことにします。"),
            ("main", "個々の曲線が平均からどれくらい散らばるかが、バリアンスの直感です。"),
        ],
    },
    {
        "id": "scene04",
        "title": "式としての分解",
        "lines": [
            ("main", "一つの入力エックスで、予測と理想の差の二乗を見ます。"),
            ("main", "そこへ平均予測を足して引くと、二つの差に分けられます。"),
            ("main", "データ集合について平均を取ると、交差項はゼロになります。"),
            ("summary", "すると、期待二乗誤差は、バイアス二乗プラス、バリアンスになります。"),
        ],
    },
    {
        "id": "scene05",
        "title": "正則化が強いとき",
        "lines": [
            ("main", "正則化ラムダが大きいと、重みはゼロに強く引き寄せられます。"),
            ("main", "どのデータ集合で学習しても、曲線は似た形になり、バリアンスは小さくなります。"),
            ("main", "しかし平均予測は、真のサインカーブから大きく外れます。"),
            ("summary", "これは、高バイアス、低バリアンスの状態です。"),
        ],
    },
    {
        "id": "scene06",
        "title": "正則化が弱いとき",
        "lines": [
            ("main", "正則化ラムダが小さいと、モデルは訓練データの細かなノイズまで追いやすくなります。"),
            ("main", "平均すれば真の曲線には近づきますが、個々の曲線は大きく揺れます。"),
            ("main", "この揺れが大きいほど、どの訓練データを引いたかに予測が敏感です。"),
            ("summary", "これは、低バイアス、高バリアンスの状態です。"),
        ],
    },
    {
        "id": "scene07",
        "title": "トレードオフ",
        "lines": [
            ("main", "ラムダを横軸にして、バイアス二乗とバリアンスを描くと、反対向きの動きが見えます。"),
            ("main", "正則化を強くすると、バリアンスは下がりますが、バイアスは上がります。"),
            ("main", "正則化を弱くすると、バイアスは下がりますが、バリアンスは上がります。"),
            ("summary", "予測性能がよいのは、その合計が小さくなる中間の複雑さです。"),
        ],
    },
    {
        "id": "scene08",
        "title": "限界と次節への橋渡し",
        "lines": [
            ("main", "バイアス、バリアンス分解は、モデル複雑さを理解するための有用な見方です。"),
            ("main", "ただし実務では、同じ分布から独立な訓練データ集合を大量に持っていることは普通ありません。"),
            ("main", "もし大量にあるなら、分解するより一つの大きな訓練データにまとめた方がよいはずです。"),
            ("summary", "そこでピーアールエムエルは次に、パラメータを平均するベイズ線形回帰へ進みます。"),
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
            wav_bytes = synthesize(base_url, str(speaker_key), str(text))
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
    parser = argparse.ArgumentParser(description="Generate PRML 3.2 narration WAV files with VOICEVOX.")
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
