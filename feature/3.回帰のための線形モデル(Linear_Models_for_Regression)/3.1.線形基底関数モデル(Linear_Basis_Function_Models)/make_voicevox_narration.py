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
        "intonation_scale": 0.96,
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

PAUSE_SECONDS = 0.28

SCENES = [
    {
        "id": "scene01",
        "title": "線形基底関数モデルとは",
        "lines": [
            ("main", "第3章では、回帰のための線形モデルを扱います。"),
            ("main", "いちばん単純な線形回帰は、入力そのものを重み付きで足します。"),
            ("main", "でも、それだけでは、曲がった関係を表す力が弱くなります。"),
            ("focus", "そこで、入力を基底関数ファイで変換してから、重み付きで足します。"),
            ("summary", "出力は入力には非線形でも、重み w には線形。この性質が、解析を扱いやすくします。"),
        ],
    },
    {
        "id": "scene02",
        "title": "基底関数を足して曲線を作る",
        "lines": [
            ("main", "基底関数は、入力の場所ごとに反応する部品だと思うと見やすくなります。"),
            ("main", "ここでは、いくつかのガウス型の山を用意します。"),
            ("main", "それぞれの山に重みを掛け、足し合わせると、全体の予測曲線ができます。"),
            ("focus", "学習で調整するのは、この山の形ではなく、主に重みの大きさです。"),
            ("summary", "固定した特徴量を作り、線形結合で予測する、という見方です。"),
        ],
    },
    {
        "id": "scene03",
        "title": "基底関数の選び方",
        "lines": [
            ("main", "基底関数の形には、いくつか代表例があります。"),
            ("main", "多項式基底は、入力全体に広がるグローバルな関数です。"),
            ("main", "ガウス基底は、特定の場所の近くで大きく反応します。"),
            ("main", "シグモイド基底は、ある位置を境に、なだらかに切り替わります。"),
            ("summary", "この節の多くの議論は、どの基底を選んでも同じ形で進みます。"),
        ],
    },
    {
        "id": "scene04",
        "title": "ガウスノイズと最小二乗",
        "lines": [
            ("main", "次に、学習の基準を確率モデルから見ます。"),
            ("main", "目標値 t は、予測 y に平均ゼロのガウスノイズが乗ったものだと仮定します。"),
            ("main", "この仮定のもとで尤度を最大にすると、残差の二乗和を小さくする問題になります。"),
            ("focus", "つまり、ガウスノイズを仮定した最尤推定と、最小二乗は同じ重みを選びます。"),
            ("summary", "確率の言葉と、二乗誤差の最小化がここでつながります。"),
        ],
    },
    {
        "id": "scene05",
        "title": "デザイン行列と正規方程式",
        "lines": [
            ("main", "データ点ごとに基底関数の値を並べると、デザイン行列ファイができます。"),
            ("main", "一行はひとつのデータ点、一列はひとつの基底関数を表します。"),
            ("main", "この行列を使うと、すべての予測値を、ファイ掛ける w とまとめて書けます。"),
            ("focus", "勾配をゼロにすると、正規方程式が出て、閉じた形の解が得られます。"),
            ("summary", "線形代数で一気に解けることが、線形モデルの大きな利点です。"),
        ],
    },
    {
        "id": "scene06",
        "title": "逐次学習",
        "lines": [
            ("main", "ただし、大きなデータを毎回まとめて解くのは重いことがあります。"),
            ("main", "データが流れてくる状況では、一点ずつ重みを更新する逐次学習が自然です。"),
            ("main", "予測が小さすぎれば、その点で反応した基底の重みを増やします。"),
            ("main", "予測が大きすぎれば、逆向きに調整します。"),
            ("summary", "この更新式が、最小平均二乗、LMS アルゴリズムです。"),
        ],
    },
    {
        "id": "scene07",
        "title": "正則化付き最小二乗",
        "lines": [
            ("main", "基底関数を増やすと、データに合わせる力は強くなります。"),
            ("main", "しかし、少ないデータでは、重みが大きくなりすぎて曲線が暴れることがあります。"),
            ("focus", "そこで、二乗誤差に、重みの大きさへの項を足します。"),
            ("main", "ラムダを大きくすると、大きすぎる重みが抑えられ、曲線はなめらかになります。"),
            ("summary", "複雑さの調整は、基底の数だけでなく、正則化係数にも移ります。"),
        ],
    },
    {
        "id": "scene08",
        "title": "q 正則化とまとめ",
        "lines": [
            ("main", "重みを抑える項は、二乗だけとは限りません。"),
            ("main", "q が二なら丸い制約になり、重みを全体的に縮めます。"),
            ("main", "q が一なら角のある制約になり、いくつかの重みがちょうどゼロになりやすくなります。"),
            ("summary", "線形基底関数モデルは、特徴量を作る部分と、線形に重みを学ぶ部分を分けて考えるモデルです。"),
            ("main", "次は、このモデルの複雑さを、バイアスとバリアンスの分解から見ていきます。"),
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
    parser = argparse.ArgumentParser(description="Generate VOICEVOX narration WAV files for PRML 3.1.")
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
