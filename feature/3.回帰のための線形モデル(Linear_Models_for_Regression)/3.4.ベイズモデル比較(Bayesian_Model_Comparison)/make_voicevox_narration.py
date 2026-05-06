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
        "title": "ベイズモデル比較の問い",
        "lines": [
            ("main", "前節では、重みそのものを分布として扱い、予測の不確かさまで見ました。"),
            ("main", "でも、基底関数の数や、モデルの種類は、まだ外側から選んでいました。"),
            ("question", "ベイズ的には、モデルそのものも確率で比べられます。"),
            ("summary", "その中心になる量が、モデルのエビデンスです。"),
            ("main", "この動画では、エビデンスが、当てはまりと複雑さをどう同時に見ているかを説明します。"),
        ],
    },
    {
        "id": "scene02",
        "title": "モデルの事後確率",
        "lines": [
            ("main", "候補モデルをエム一、エム二、エム三のように並べます。"),
            ("main", "データを見たあとに知りたいのは、それぞれのモデルがどれくらいもっともらしいかです。"),
            ("main", "ベイズの式では、モデルの事後確率は、モデル事前確率と、データがそのモデルから出る確率に比例します。"),
            ("summary", "この、データがそのモデルから出る確率が、エビデンス、または周辺尤度です。"),
        ],
    },
    {
        "id": "scene03",
        "title": "重みを積分して消す",
        "lines": [
            ("main", "ひとつのモデルの中にも、重みベクトル w という未知量があります。"),
            ("main", "エビデンスでは、特定の重みだけを選びません。"),
            ("main", "重みの事前分布と、データへの尤度を掛け合わせ、重み空間全体で足し合わせます。"),
            ("summary", "つまり、モデルの中で可能な説明を全部集めた、モデル全体のデータ生成確率です。"),
        ],
    },
    {
        "id": "scene04",
        "title": "当てはまりとオッカム因子",
        "lines": [
            ("main", "複雑なモデルは、うまく合う重みを見つけやすいので、最大尤度だけを見ると有利です。"),
            ("main", "一方で、複雑なモデルの重み空間は広く、その多くはデータをうまく説明しません。"),
            ("main", "エビデンスは、よく当たる領域の高さだけでなく、その領域が事前分布の中でどれくらい占めるかも見ます。"),
            ("summary", "この自動的な複雑さの割引を、オッカム因子として見ることができます。"),
        ],
    },
    {
        "id": "scene05",
        "title": "多項式モデルで比べる",
        "lines": [
            ("main", "多項式の次数で考えると、低すぎる次数は、そもそもデータの形を表せません。"),
            ("main", "高すぎる次数は、訓練点には合わせやすいものの、使える重みの範囲が広すぎます。"),
            ("main", "エビデンスは、中くらいの次数を高く評価することがあります。"),
            ("summary", "これは検証データを分けずに、ベイズの式の中で複雑さを評価している、と見られます。"),
        ],
    },
    {
        "id": "scene06",
        "title": "モデル同士の比",
        "lines": [
            ("main", "二つのモデルを直接比べるときは、事後確率の比を見ます。"),
            ("main", "その比は、事前確率の比と、エビデンスの比に分かれます。"),
            ("main", "事前にどちらも同じくらいありそうだと思うなら、勝敗を決めるのはエビデンス比です。"),
            ("summary", "データが増えるほど、根拠のあるモデルへ事後確率が集中していきます。"),
        ],
    },
    {
        "id": "scene07",
        "title": "モデル平均",
        "lines": [
            ("main", "モデル比較は、必ずひとつだけを選ぶためのものではありません。"),
            ("main", "ベイズ的には、各モデルの予測を、モデルの事後確率で重み付けして足し合わせることもできます。"),
            ("main", "これがベイズモデル平均です。"),
            ("summary", "モデルがまだはっきり決まらないとき、その不確かさを予測にも残せます。"),
        ],
    },
    {
        "id": "scene08",
        "title": "次節への橋渡し",
        "lines": [
            ("main", "ベイズモデル比較の要点は、モデルごとに、重みを積分したデータ生成確率を比べることです。"),
            ("main", "エビデンスは、当てはまりの良さと、不要に広い自由度への割引を同時に含みます。"),
            ("main", "ただし、この積分をそのまま計算するのは、一般には簡単ではありません。"),
            ("summary", "そこで次は、エビデンスを実際に近似して使う、エビデンス近似へ進みます。"),
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
    query["postPhonemeLength"] = 0.20
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
    parser = argparse.ArgumentParser(description="Generate VOICEVOX narration WAV files for PRML 3.4.")
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
