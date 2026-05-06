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
        "title": "固定基底関数とは",
        "lines": [
            ("main", "線形基底関数モデルでは、入力をいくつかの基底関数ファイに通してから、重み付きで足し合わせます。"),
            ("main", "学習で調整するのは主に重みで、基底関数の位置や幅は先に決めておくことが多いです。"),
            ("focus", "この固定した特徴量という選び方が、計算を扱いやすくします。"),
            ("summary", "しかし同時に、表現できる形は、最初に置いた基底関数に強く依存します。"),
        ],
    },
    {
        "id": "scene02",
        "title": "幅のトレードオフ",
        "lines": [
            ("main", "ガウス型の局所基底を使うと、幅の選び方だけでもトレードオフが出ます。"),
            ("main", "狭すぎる基底は、近くの点だけに強く反応し、点ごとのゆらぎを拾いやすくなります。"),
            ("main", "広すぎる基底は、遠くの点までまとめるので、細かい変化をならしてしまいます。"),
            ("focus", "ほどよい幅は、データの密度や関数の変化の速さに依存します。"),
            ("summary", "固定基底では、この調整を学習の前に決めなければなりません。"),
        ],
    },
    {
        "id": "scene03",
        "title": "高次元で増える基底数",
        "lines": [
            ("main", "入力が一つなら、区間上に基底を並べるイメージは分かりやすいです。"),
            ("main", "でも入力が二つになると、面の上に格子として並べる必要があります。"),
            ("main", "三つなら立体の格子になります。"),
            ("focus", "各次元にケー個ずつ置くと、必要数はケーのディー乗です。"),
            ("summary", "次元が増えると、固定した局所基底はすぐに数が増えすぎます。"),
        ],
    },
    {
        "id": "scene04",
        "title": "空白にも基底を置いてしまう",
        "lines": [
            ("main", "さらに、実際のデータは入力空間全体に均等にあるとは限りません。"),
            ("main", "低い次元の曲線や面の近くにだけ集まることもあります。"),
            ("main", "固定グリッドは、データがほとんどない場所にも同じように基底を置きます。"),
            ("focus", "高次元では、この空白部分がとても大きくなります。"),
            ("summary", "必要な場所に細かく、不要な場所には少なく、という配分が難しくなります。"),
        ],
    },
    {
        "id": "scene05",
        "title": "基底の形が合わない問題",
        "lines": [
            ("main", "もう一つの問題は、基底の形がデータの形に合わない場合です。"),
            ("main", "固定された少数の基底では、必要な場所に部品がないことがあります。"),
            ("main", "重みだけを変えても、部品の位置や幅そのものは動きません。"),
            ("focus", "そのため、同じデータを表すのに多くの基底が必要になったり、なめらかさを失ったりします。"),
            ("summary", "固定基底の限界は、重みの学習ではなく、特徴量の設計にあります。"),
        ],
    },
    {
        "id": "scene06",
        "title": "データ適応的な基底へ",
        "lines": [
            ("main", "この限界を超える自然な方向は、基底関数をデータに合わせて動かすことです。"),
            ("main", "必要な場所に中心を寄せる。必要なスケールに幅を変える。"),
            ("main", "あるいは、特徴量そのものを学習する。"),
            ("focus", "線形モデルの扱いやすさを残しながら、特徴量をより賢く選ぶ発想です。"),
            ("summary", "ここから、カーネル法、スパースな基底選択、ニューラルネットワークへの動機が見えてきます。"),
        ],
    },
    {
        "id": "scene07",
        "title": "まとめ",
        "lines": [
            ("main", "固定基底関数は、線形回帰を非線形な関係へ広げる便利な道具です。"),
            ("main", "しかし、局所基底では幅の調整が難しく、高次元では必要数が急に増えます。"),
            ("main", "また、データのない場所にも基底を置き、データの形に合わないことがあります。"),
            ("focus", "つまり限界は、重みをどう解くかではなく、どんな特徴量を用意するかにあります。"),
            ("summary", "次の発想は、特徴量を固定せず、データに合わせて扱うことです。"),
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
    parser = argparse.ArgumentParser(description="Generate VOICEVOX narration WAV files for PRML 3.6.")
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
