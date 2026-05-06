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
        "title": "エビデンス近似の位置づけ",
        "lines": [
            ("main", "ベイズ線形回帰では、重みダブリューを分布として扱いました。"),
            ("main", "本当は、正則化の強さアルファや、ノイズ精度ベータにも分布を置き、全部を積分したいところです。"),
            ("main", "しかし、重みとハイパーパラメータをすべて積分するのは、一般には扱いにくくなります。"),
            ("summary", "そこで、重みは積分し、アルファとベータは、データのエビデンスが最大になる値で代表させます。"),
            ("main", "この近似を、エビデンス近似、または経験ベイズと呼びます。"),
        ],
    },
    {
        "id": "scene02",
        "title": "エビデンスとは",
        "lines": [
            ("question", "エビデンスとは、何を測っているのでしょうか。"),
            ("main", "式では、尤度と事前分布を掛けて、重みダブリューで積分した量です。"),
            ("main", "よく当てはまる重みがあるだけでは足りません。"),
            ("main", "事前分布の中で、どれくらい広い重みの領域がデータを説明できるかも効きます。"),
            ("summary", "つまりエビデンスは、当てはまりの良さと、複雑さの使いすぎを同時に見る尺度です。"),
        ],
    },
    {
        "id": "scene03",
        "title": "log evidence の分解",
        "lines": [
            ("main", "線形ガウスモデルでは、重みに関する積分を解析的に計算できます。"),
            ("main", "対数エビデンスは、アルファとベータの正規化項、データ誤差、重みの大きさ、そしてヘッセ行列の行列式に分かれます。"),
            ("main", "データ誤差を小さくするほど良い一方で、重みを細かく調整しすぎると、行列式の項が複雑さとして効きます。"),
            ("summary", "このつり合いが、検証データなしで正則化の強さを選ぶ仕組みです。"),
        ],
    },
    {
        "id": "scene04",
        "title": "alpha を evidence で選ぶ",
        "lines": [
            ("main", "アルファは、重みをゼロの近くに保つ強さです。"),
            ("main", "アルファが大きすぎると、曲線は硬くなり、データを追えません。"),
            ("main", "小さすぎると、重みが自由になりすぎ、細かな揺れまで拾います。"),
            ("summary", "エビデンスをアルファの関数として見ると、中間の値で山ができます。"),
            ("main", "この山の頂点が、訓練データから選ばれるアルファです。"),
        ],
    },
    {
        "id": "scene05",
        "title": "有効パラメータ数 gamma",
        "lines": [
            ("main", "エビデンス近似では、ガンマという量が現れます。"),
            ("main", "各方向について、データが作る曲率ラムダアイがアルファより大きければ、その方向の重みはデータでよく決まります。"),
            ("main", "逆に、曲率が小さい方向は、事前分布に押し戻され、ほとんど使われません。"),
            ("summary", "ガンマは、ゼロからエムまでの間で動く、有効なパラメータ数です。"),
        ],
    },
    {
        "id": "scene06",
        "title": "beta の更新",
        "lines": [
            ("main", "ベータは、観測ノイズの精度です。逆数はノイズ分散に対応します。"),
            ("main", "最尤推定なら、残差二乗和をデータ数エヌで割って分散を見積もります。"),
            ("main", "ベイズ的な更新では、分母がエヌではなく、エヌマイナスガンマになります。"),
            ("summary", "データで決めた有効パラメータ数の分だけ、自由度を差し引くと考えられます。"),
        ],
    },
    {
        "id": "scene07",
        "title": "再推定の反復",
        "lines": [
            ("main", "実際には、アルファもベータも、ガンマや事後平均エムエヌに依存します。"),
            ("main", "そこで、まず初期値を置き、重みの事後平均を計算します。"),
            ("main", "そこからガンマを求め、アルファとベータを更新します。"),
            ("main", "この手順を繰り返すと、エビデンスの高い値へ近づいていきます。"),
            ("summary", "交差検証の候補探索ではなく、訓練データの周辺尤度を直接押し上げる流れです。"),
        ],
    },
    {
        "id": "scene08",
        "title": "まとめ",
        "lines": [
            ("main", "エビデンス近似は、完全ベイズよりは近似的ですが、重みを一点に固定しません。"),
            ("main", "重みを積分したうえで、ハイパーパラメータを周辺尤度で選びます。"),
            ("main", "その結果、アルファとベータ、つまり正則化とノイズを、訓練データから同時に推定できます。"),
            ("summary", "中心にある直感は、よく当てはまり、しかも必要以上に自由すぎない説明を選ぶ、ということです。"),
            ("main", "次は、固定された基底関数を使うこと自体の限界へ進みます。"),
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
    tmp_output_path = OUTPUT_DIR / f"_{scene_id}.wav"
    try:
        with wave.open(str(tmp_output_path), "wb") as output:
            for speaker_key, text in scene["lines"]:  # type: ignore[index]
                wav_bytes = synthesize(base_url, str(speaker_key), str(text))
                expected_params = append_wav_bytes(output, wav_bytes, expected_params)
                append_silence(output, expected_params, PAUSE_SECONDS)
                time.sleep(0.08)
        tmp_output_path.replace(output_path)
    except Exception:
        tmp_output_path.unlink(missing_ok=True)
        raise

    with wave.open(str(output_path), "rb") as generated:
        duration = generated.getnframes() / generated.getframerate()
    return {"id": scene_id, "title": scene["title"], "path": str(output_path), "duration": duration, "skipped": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VOICEVOX narration WAV files for PRML 3.5.")
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
