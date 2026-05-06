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

PAUSE_SECONDS = 0.3

SCENES = [
    {
        "id": "scene01",
        "title": "パラメータで形を決める限界",
        "lines": [
            ("main", "これまでの章では、ベルヌーイ分布、ガウス分布、指数型分布族のように、先に分布の形を決めてから、少数のパラメータをデータで調整してきました。"),
            ("main", "この考え方は強力ですが、形の決め打ちが外れることがあります。"),
            ("main", "たとえば、データが二つの山を持っているのに、一つのガウス分布だけで説明しようとすると、中央の谷まで高く見積もってしまいます。"),
            ("main", "そこで、分布の形をなるべく先に固定しない方法を考えます。"),
            ("summary", "これがノンパラメトリック手法の入口です。"),
        ],
    },
    {
        "id": "scene02",
        "title": "ヒストグラムは最初の密度推定",
        "lines": [
            ("main", "一番身近なノンパラメトリックな密度推定は、ヒストグラムです。"),
            ("main", "横軸を小さな区間に分け、その中に入った点の数を数えます。"),
            ("main", "密度として使うには、個数を全データ数と区間の幅で割ります。"),
            ("main", "ビン幅が小さすぎると、偶然のばらつきまでギザギザに拾います。"),
            ("main", "逆に大きすぎると、二つの山が一つにつぶれてしまいます。"),
            ("summary", "よい幅は、小さすぎず、大きすぎない中間にあります。"),
        ],
    },
    {
        "id": "scene03",
        "title": "局所近傍で数えるという共通原理",
        "lines": [
            ("main", "ヒストグラムから分かる大事なことは、密度を知りたい場所の近くで点を数える、ということです。"),
            ("main", "場所エックスのまわりに、小さな領域アールを置きます。"),
            ("main", "そこにケー個の点が入っていて、領域の体積がブイなら、密度は、ケー割るエヌブイ、と見積もれます。"),
            ("main", "ただし、領域は小さいほど局所的ですが、点が少なすぎると不安定です。"),
            ("main", "大きくすると安定しますが、細かい構造は見えにくくなります。"),
            ("summary", "この矛盾が、この節全体の調整つまみになります。"),
        ],
    },
    {
        "id": "scene04",
        "title": "カーネル密度推定",
        "lines": [
            ("main", "カーネル密度推定では、体積ブイ、つまり幅エイチを先に固定します。"),
            ("main", "そして、各データ点の上に小さな山を一つずつ置きます。"),
            ("main", "ガウスカーネルを使うなら、データ点の数だけガウスの山を置き、全部を足して、最後にエヌで割ります。"),
            ("main", "エイチが小さいと、一つ一つの点を強く覚えた、針のような密度になります。"),
            ("main", "エイチが大きいと、全体は滑らかですが、二つの山の違いが薄れます。"),
            ("summary", "ここでも、幅エイチは小さすぎず大きすぎない値を選ぶ必要があります。"),
        ],
    },
    {
        "id": "scene05",
        "title": "K近傍密度推定",
        "lines": [
            ("main", "ケー近傍法では、逆にケーを先に固定します。"),
            ("main", "場所エックスのまわりの球を、ケー個の点が入るまで広げます。"),
            ("main", "密な場所なら、半径は小さくて済みます。"),
            ("main", "まばらな場所なら、同じケー個を集めるために半径が大きくなります。"),
            ("main", "つまりケー近傍密度推定は、場所ごとに近傍の大きさを変える方法です。"),
            ("summary", "ケーが小さすぎると不安定になり、大きすぎるとやはり滑らかになりすぎます。"),
        ],
    },
    {
        "id": "scene06",
        "title": "K近傍法で分類する",
        "lines": [
            ("main", "ケー近傍の考え方は、分類にもそのまま使えます。"),
            ("main", "新しい点の近くにあるケー個の訓練データを探します。"),
            ("main", "その中で一番多いクラスを、予測結果にします。"),
            ("main", "ピーアールエムエルの導出では、クラスごとに密度を見積もり、ベイズの定理を使うと、結局ケー個の中の多数決になります。"),
            ("main", "ケーが一なら最近傍の一点だけで決めます。"),
            ("summary", "ケーを大きくすると、境界は滑らかになりますが、細かな構造は消えやすくなります。"),
        ],
    },
    {
        "id": "scene07",
        "title": "強みと弱み",
        "lines": [
            ("main", "ノンパラメトリック手法の強みは、最初から一つの決まった形に押し込めないことです。"),
            ("main", "山が二つあれば二つの山として見えますし、場所によって密度が変わる様子も表せます。"),
            ("main", "一方で、カーネル密度推定もケー近傍法も、訓練データそのものを保持して、評価時に参照します。"),
            ("main", "そのため、データが大きいほど計算が重くなります。"),
            ("main", "柔軟さを持ちつつ、モデルの複雑さをデータ数から切り離して制御したい。"),
            ("summary", "この課題が、後の章で出てくる混合モデルやカーネル法へつながっていきます。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 2.5 narration WAV files with VOICEVOX.")
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
