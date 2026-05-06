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

PAUSE_SECONDS = 0.3

SCENES = [
    {
        "id": "scene01",
        "title": "勾配の次に曲率を見る",
        "lines": [
            ("main", "ニューラルネットワークの訓練では、誤差関数を重みで微分した勾配を使って、重みを動かします。"),
            ("main", "ヘッセ行列は、その勾配がさらにどう変わるかを集めた、二階微分の行列です。"),
            ("main", "一つ一つの成分は、重みアイと重みジェイを少し変えたとき、誤差の曲がり方がどう結びつくかを表します。"),
            ("main", "そのため、単なる傾きではなく、誤差面の細長さ、平坦さ、交差する曲率まで見ることができます。"),
            ("summary", "この節では、ヘッセ行列を、曲率を読む地図として見ていきます。"),
        ],
    },
    {
        "id": "scene02",
        "title": "局所二次近似",
        "lines": [
            ("main", "まず、ある重みの近くで誤差関数をテイラー展開します。"),
            ("main", "一次の項には勾配が入り、二次の項にはヘッセ行列が入ります。"),
            ("main", "最小点の近くでは勾配がゼロなので、局所的な形は二次の項がほぼ決めます。"),
            ("main", "つまりヘッセ行列は、最小点の近くの誤差面を、楕円形のボウルとして近似するときの形そのものです。"),
            ("summary", "勾配が場所を動かす量なら、ヘッセ行列はその周りの地形を説明する量です。"),
        ],
    },
    {
        "id": "scene03",
        "title": "固有値と正定値性",
        "lines": [
            ("main", "ヘッセ行列を固有値分解すると、固有ベクトルが曲率の主な向きを表します。"),
            ("main", "対応する固有値が大きい方向では、少し動いただけで誤差が急に増えます。"),
            ("main", "固有値が小さい方向では、等高線が長く伸び、誤差面は平坦に見えます。"),
            ("main", "すべての固有値が正なら、その点のまわりではどの方向に動いても誤差が増えるので、局所最小と判断できます。"),
            ("summary", "負の固有値やゼロに近い固有値は、鞍点や平坦な谷を疑うサインになります。"),
        ],
    },
    {
        "id": "scene04",
        "title": "ニューラルネットワークでの役割",
        "lines": [
            ("main", "ヘッセ行列は、ニューラルネットワークの訓練と解析のいくつかの場面で重要です。"),
            ("main", "二次の最適化では、曲率を使って、どの方向にどれくらい進むかを調整します。"),
            ("main", "訓練データが少し変わったときの再学習の近似にも使えます。"),
            ("main", "さらに、逆ヘッセ行列は、重みを削っても影響が小さいかを調べる刈り込みの考え方に使われます。"),
            ("main", "ベイズニューラルネットワークでは、ラプラス近似、予測分布、モデル証拠にも現れます。"),
            ("summary", "ただし重みが W 個あると、ヘッセ行列は W かける W の大きさになります。"),
        ],
    },
    {
        "id": "scene05",
        "title": "近似の考え方",
        "lines": [
            ("main", "大きなヘッセ行列をそのまま扱うのは重いので、いくつかの近似が使われます。"),
            ("main", "対角近似では、非対角成分をゼロにして、重み同士の結びつきを捨てます。逆行列は簡単ですが、実際のヘッセ行列は強く非対角なことがあります。"),
            ("main", "外積近似では、各データ点の一階微分ベクトルの外積を足し合わせます。"),
            ("main", "これは、二乗誤差で、訓練後の残差が小さいときに使いやすい近似です。"),
            ("summary", "近似を使うときは、計算を軽くする代わりに、どの項を無視しているかを確認します。"),
        ],
    },
    {
        "id": "scene06",
        "title": "逆ヘッセ行列と有限差分",
        "lines": [
            ("main", "外積近似を使うと、データ点を一つずつ足しながら、逆ヘッセ行列を更新できます。"),
            ("main", "これはウッドベリー恒等式の形で、すでにある逆行列に、新しいデータ点の情報を吸収していきます。"),
            ("main", "一方、二階微分は有限差分でも計算できます。"),
            ("main", "ただし、すべての重みの組を直接差分で見ると計算量が大きくなります。"),
            ("summary", "実用上、有限差分は本番の計算方法というより、二階 backprop の実装確認に向いています。"),
        ],
    },
    {
        "id": "scene07",
        "title": "正確な評価とヘッセベクトル積",
        "lines": [
            ("main", "ヘッセ行列は、backpropagation を二階微分へ拡張することで、正確に評価できます。"),
            ("main", "ただし、多くの応用では、行列全体ではなく、ヘッセ行列とあるベクトルの積だけが欲しいことがあります。"),
            ("main", "このときは、R オペレーターを使って、通常の forward と backward に対応する追加の量を流します。"),
            ("main", "すると、全ヘッセ行列を保存せずに、ヘッセベクトル積を効率よく得られます。"),
            ("summary", "ヘッセ行列は、曲率を読む道具です。必要な精度と計算量に合わせて、近似、正確な評価、ベクトル積を選びます。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 5.4 narration WAV files with VOICEVOX.")
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
