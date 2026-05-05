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
        "intonation_scale": 0.94,
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
        "title": "オープニング",
        "lines": [
            ("main", "今回は、ピーアールエムエル一章四節、次元の呪いを見ます。"),
            ("main", "ここでいう次元とは、入力を表す数字の個数です。"),
            ("main", "一つの数字なら線の上、二つなら平面の上に点を置けます。"),
            ("main", "でも、十個、百個と増えると、人間には見えない向きに空間が広がっていきます。"),
            ("summary", "この広がり方が、機械学習を急に難しくします。"),
        ],
    },
    {
        "id": "scene02",
        "title": "格子数の指数増加",
        "lines": [
            ("main", "まず、各座標を十個の区間に分けるとします。"),
            ("main", "一次元なら箱は十個です。二次元なら、十かける十で百個になります。"),
            ("main", "三次元なら千個。ここまでは、まだ想像できます。"),
            ("main", "ところが十次元では、十の十乗、つまり百億個の箱になります。"),
            ("main", "同じ細かさで空間を見ようとするだけで、必要な観察量が指数的に増えてしまいます。"),
        ],
    },
    {
        "id": "scene03",
        "title": "局所領域が広くなる",
        "lines": [
            ("main", "高次元で、近い場所だけを見る、という感覚も変わります。"),
            ("main", "中心のまわりに、全体の十パーセントの体積を集めたいとします。"),
            ("main", "一次元なら、軸の幅は十パーセントで済みます。"),
            ("main", "でも十次元では、一つ一つの軸で約七十九パーセントもの幅を使う必要があります。"),
            ("question", "十パーセントだけ見たいのに、各座標ではほとんど全体を見るんですか？"),
            ("main", "そうです。高次元では、体積は座標を掛け合わせて決まるからです。"),
        ],
    },
    {
        "id": "scene04",
        "title": "境界近くの体積",
        "lines": [
            ("main", "もう一つ大事なのは、体積が境界の近くに寄りやすいことです。"),
            ("main", "各軸の両端から十パーセントを境界帯として取り除くと、中心に残る幅は八十パーセントです。"),
            ("main", "二次元なら、中心に残る面積は六十四パーセントです。"),
            ("main", "でも十次元では約十パーセント、二十次元では約一パーセントまで減ります。"),
            ("summary", "高次元の多くの点は、どこかの座標で端の近くにいます。"),
        ],
    },
    {
        "id": "scene05",
        "title": "近傍点が遠くなる",
        "lines": [
            ("main", "次は、近くの点を探す場面です。"),
            ("main", "二次元なら、点をたくさん置けば、中心の近くに別の点が見つかりそうです。"),
            ("main", "しかし同じ三百点でも、次元が増えると、中心から一番近い点までの距離は大きくなります。"),
            ("main", "近傍法のように、近いサンプルを頼りにする方法は、この影響を強く受けます。"),
            ("main", "データが多そうに見えても、高次元空間の中ではまばらになっているのです。"),
        ],
    },
    {
        "id": "scene06",
        "title": "空の箱だらけになる",
        "lines": [
            ("main", "空間を箱に分けて、各箱で何が起きるか覚える方法を考えてみます。"),
            ("main", "各軸を五分割するだけなら、二次元では二十五個の箱です。"),
            ("main", "でも六次元では一万五千個を超え、十次元では約九百七十六万個になります。"),
            ("main", "千個のデータがあっても、ほとんどの箱は一度も観察されません。"),
            ("summary", "高次元では、表を埋めるような学習はすぐ限界にぶつかります。"),
        ],
    },
    {
        "id": "scene07",
        "title": "何が助けになるか",
        "lines": [
            ("main", "では、どうすればよいのでしょうか。"),
            ("main", "答えは、空間をそのまま全部埋めようとしないことです。"),
            ("main", "入力の中で本当に効く座標は一部だけかもしれません。"),
            ("main", "近い入力は近い出力を持つ、という滑らかさの仮定も使えます。"),
            ("main", "また、正則化によって、複雑すぎる関数を選びにくくできます。"),
            ("summary", "データ量だけでなく、よい仮定が汎化を支えます。"),
        ],
    },
    {
        "id": "scene08",
        "title": "まとめ",
        "lines": [
            ("main", "次元の呪いとは、次元が増えると空間が急激に広くなり、データがまばらになる問題です。"),
            ("main", "同じ細かさを保つには、必要な箱やデータ数が指数的に増えます。"),
            ("main", "さらに、体積は境界側に寄り、近い点も見つけにくくなります。"),
            ("main", "だから、機械学習では、特徴選択、次元削減、正則化、モデルの仮定が重要になります。"),
            ("main", "次の節では、予測結果をどう判断や行動につなげるか、決定理論へ進みます。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 1.4 narration WAV files with VOICEVOX.")
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
