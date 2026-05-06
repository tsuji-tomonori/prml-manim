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

PAUSE_SECONDS = 0.28

SCENES = [
    {
        "id": "scene01",
        "title": "点推定から分布へ",
        "lines": [
            ("main", "三章三節では、線形回帰をベイズ的に見直します。"),
            ("main", "これまでの最小二乗や正則化では、重みベクトルをひとつの値として求めました。"),
            ("main", "ベイズ線形回帰では、重みそのものを確率分布として扱います。"),
            ("question", "つまり、どの直線が正しいかを一つに決めるのではありません。"),
            ("summary", "データを見たあと、ありそうな直線の分布として答えるのが出発点です。"),
        ],
    },
    {
        "id": "scene02",
        "title": "事前分布と重み空間",
        "lines": [
            ("main", "まず、データを見る前の重みの分布を置きます。"),
            ("main", "ここでは切片と傾きの二つだけを考えると、重み空間は平面になります。"),
            ("main", "原点の近くを好むガウス分布を置くと、大きすぎる重みは最初から少し起こりにくくなります。"),
            ("main", "この広がりを決めるのが、事前分布の精度アルファです。"),
            ("summary", "アルファが大きいほど、重みはゼロの近くに強く集まります。"),
        ],
    },
    {
        "id": "scene03",
        "title": "尤度が重みを絞る",
        "lines": [
            ("main", "観測点が一つ入ると、その点を説明しやすい重みに高い尤度が付きます。"),
            ("main", "重み空間で見ると、点一つは細長い帯のような制約になります。"),
            ("main", "事前分布に、この尤度を掛けて正規化すると、事後分布になります。"),
            ("main", "線形ガウスモデルでは、事後分布もガウス分布のままです。"),
            ("summary", "形がガウスのまま更新できるので、平均と共分散だけで追跡できます。"),
        ],
    },
    {
        "id": "scene04",
        "title": "逐次更新",
        "lines": [
            ("main", "データを一つずつ追加すると、事後分布は少しずつ狭くなります。"),
            ("main", "最初は多くの直線がありえますが、観測が増えるほど候補は絞られます。"),
            ("main", "重み空間の楕円が小さくなることと、グラフ上の直線のばらつきが小さくなることは同じ現象です。"),
            ("question", "ここで大事なのは、一本の最良直線だけを追っていないことです。"),
            ("summary", "不確実性がどのくらい残っているかも、学習結果の一部として残ります。"),
        ],
    },
    {
        "id": "scene05",
        "title": "予測分布",
        "lines": [
            ("main", "新しい入力エックススターでの予測も、ひとつの値ではなく分布になります。"),
            ("main", "平均は、事後平均の重みで作る通常の回帰曲線です。"),
            ("main", "分散には二つの成分があります。観測ノイズと、重みがまだ不確かなことによる揺らぎです。"),
            ("main", "データが多い場所では帯が細く、離れた場所では帯が太くなります。"),
            ("summary", "ベイズ予測は、当たりそうな値と同時に、どのくらい自信があるかを返します。"),
        ],
    },
    {
        "id": "scene06",
        "title": "アルファとベータ",
        "lines": [
            ("main", "式に出てくるアルファとベータは、不確実性のつまみです。"),
            ("main", "アルファは事前分布の精度で、重みをどれだけゼロ近くへ寄せるかを決めます。"),
            ("main", "ベータは観測ノイズの精度で、データ点をどれだけ信用するかを決めます。"),
            ("main", "ベータが大きいほどノイズは小さいとみなし、予測の帯は細くなります。"),
            ("summary", "この二つの値は、後のエビデンス近似でデータから選ぶ対象になります。"),
        ],
    },
    {
        "id": "scene07",
        "title": "3.4 への橋渡し",
        "lines": [
            ("main", "ベイズ線形回帰は、点推定を分布推定へ広げました。"),
            ("main", "重みの不確実性を残すことで、予測分布と信頼の幅を自然に計算できます。"),
            ("main", "さらに、分布全体を積分すると、モデルそのもののもっともらしさを比べられます。"),
            ("summary", "次の三章四節では、この考えを使って、ベイズモデル比較へ進みます。"),
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
    query["postPhonemeLength"] = 0.18
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


def synthesize_scene(base_url: str, scene: dict[str, object], overwrite: bool) -> dict[str, object]:
    scene_id = str(scene["id"])
    output_path = OUTPUT_DIR / f"{scene_id}.wav"
    if output_path.exists() and not overwrite:
        with wave.open(str(output_path), "rb") as existing:
            duration = existing.getnframes() / existing.getframerate()
        return {"id": scene_id, "title": scene["title"], "path": str(output_path), "duration": duration, "skipped": True}

    expected_params: tuple[int, int, int] | None = None
    with wave.open(str(output_path), "wb") as output:
        for speaker_key, text in scene["lines"]:  # type: ignore[index]
            wav_bytes = synthesize(base_url, str(speaker_key), str(text))
            expected_params = append_wav_bytes(output, wav_bytes, expected_params)
            append_silence(output, expected_params, PAUSE_SECONDS)
            time.sleep(0.08)

    with wave.open(str(output_path), "rb") as generated:
        duration = generated.getnframes() / generated.getframerate()
    return {"id": scene_id, "title": scene["title"], "path": str(output_path), "duration": duration, "skipped": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VOICEVOX narration WAV files for PRML 3.3.")
    parser.add_argument("--base-url", default="http://127.0.0.1:50021", help="VOICEVOX Engine base URL")
    parser.add_argument("--from-scene", default=None, help="Resume from a scene id, e.g. scene04")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing WAV files")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = SCENES
    if args.from_scene:
        scene_ids = [scene["id"] for scene in SCENES]
        if args.from_scene not in scene_ids:
            raise SystemExit(f"Unknown scene id: {args.from_scene}")
        selected = SCENES[scene_ids.index(args.from_scene) :]

    manifest: list[dict[str, object]] = []
    manifest_path = OUTPUT_DIR / "manifest.json"
    if manifest_path.exists() and args.from_scene:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected_ids = {scene["id"] for scene in selected}
        manifest = [item for item in manifest if item.get("id") not in selected_ids]

    for scene in selected:
        result = synthesize_scene(args.base_url, scene, overwrite=args.overwrite)
        status = "skip" if result["skipped"] else "generated"
        print(f"{status}: {result['id']} {result['duration']:.2f}s")
        manifest.append(result)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
