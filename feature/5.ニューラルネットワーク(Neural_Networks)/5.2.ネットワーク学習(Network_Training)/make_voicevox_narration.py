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
        "speed_scale": 1.04,
        "intonation_scale": 0.9,
    },
    "summary": {
        "label": "VOICEVOX:WhiteCUL",
        "id": 23,
        "speed_scale": 1.02,
        "intonation_scale": 0.88,
    },
}

PAUSE_SECONDS = 0.28

SCENES = [
    {
        "id": "scene01",
        "title": "学習は誤差関数を下げること",
        "lines": [
            ("main", "ニューラルネットワークは、入力から出力を作る非線形な関数です。"),
            ("main", "ネットワーク学習では、この関数の形を決める重みベクトル w を、訓練データに合わせて調整します。"),
            ("main", "回帰では、出力 y と目標 t の差を二乗して足し合わせる誤差関数を使えます。"),
            ("main", "確率モデルとして見ると、ガウス雑音を仮定した最大尤度推定が、この二乗和誤差の最小化に対応します。"),
            ("summary", "つまり学習とは、重み空間の上で誤差 E w が小さくなる場所を探す問題です。"),
        ],
    },
    {
        "id": "scene02",
        "title": "出力と誤差関数は組で選ぶ",
        "lines": [
            ("main", "PRML 5.2 では、問題の種類ごとに、出力ユニットと誤差関数を自然な組として選びます。"),
            ("main", "実数値を予測する回帰では、線形出力と二乗和誤差を使います。"),
            ("main", "二値分類では、シグモイド出力を確率として読み、ベルヌーイ分布の負の対数尤度、つまりクロスエントロピーを使います。"),
            ("main", "多クラス分類では、ソフトマックス出力で確率の合計を一にそろえ、対応する多クラスクロスエントロピーを使います。"),
            ("summary", "この組み合わせにすると、出力層の微分が y マイナス t という扱いやすい形になります。"),
        ],
    },
    {
        "id": "scene03",
        "title": "重み空間の地形を下る",
        "lines": [
            ("main", "選んだ誤差関数は、重み w の関数として、重み空間の上に山や谷を作ります。"),
            ("main", "勾配ナブラ E は、その地点で誤差が最も増える向きを指します。"),
            ("main", "したがって、マイナス勾配の方向へ小さく進むと、誤差を下げられます。"),
            ("main", "誤差が十分小さい場所では、勾配がゼロに近くなります。"),
            ("summary", "ただし非線形ネットワークの誤差面は非凸なので、谷は一つとは限りません。"),
        ],
    },
    {
        "id": "scene04",
        "title": "局所二次近似とヘッセ行列",
        "lines": [
            ("main", "ある点の近くでは、複雑な誤差面も、二次関数のように近似できます。"),
            ("main", "この近似では、勾配が傾きを表し、ヘッセ行列が曲がり方を表します。"),
            ("main", "最小点のまわりでは、等高線は楕円になり、その軸はヘッセ行列の固有ベクトルに沿います。"),
            ("main", "固有値が大きい方向は急で、固有値が小さい方向はゆるやかです。"),
            ("summary", "この曲がり方の違いが、単純な勾配降下をジグザグさせる原因になります。"),
        ],
    },
    {
        "id": "scene05",
        "title": "勾配情報を使う理由",
        "lines": [
            ("main", "重みが W 個あるとき、関数値を一つ測るだけでは、どちらへ進むべきかの情報は少ししか得られません。"),
            ("main", "一方で、勾配を一回計算すると、W 個の方向成分が同時に得られます。"),
            ("main", "バックプロパゲーションを使えば、この勾配を重み数に比例する計算量で効率よく求められます。"),
            ("main", "そのため実用的なニューラルネットワーク学習は、勾配情報を中心に組み立てられます。"),
            ("summary", "次の節の誤差逆伝播は、この勾配をどう計算するかの話です。"),
        ],
    },
    {
        "id": "scene06",
        "title": "バッチ更新とオンライン更新",
        "lines": [
            ("main", "最も単純な更新は、現在の重みから、学習率 eta 倍の勾配を引く勾配降下です。"),
            ("main", "全データの誤差をまとめて勾配にする方法を、バッチ法と呼びます。"),
            ("main", "一方、データ点一つ、または小さなミニバッチごとに更新する方法は、オンライン、または確率的勾配降下と呼ばれます。"),
            ("main", "オンライン更新は大きなデータで重複を扱いやすく、全体の局所最小から抜け出す揺らぎも持ちます。"),
            ("summary", "ネットワーク学習は、誤差関数、勾配、学習率、データの使い方を組み合わせて進める数値最適化です。"),
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
    query["postPhonemeLength"] = 0.2
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
    parser = argparse.ArgumentParser(description="Generate PRML 5.2 narration WAV files with VOICEVOX.")
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
