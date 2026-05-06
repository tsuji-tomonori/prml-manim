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

PAUSE_SECONDS = 0.32

SCENES = [
    {
        "id": "scene01",
        "title": "逆問題と多峰性",
        "lines": [
            ("main", "五章六節では、混合密度ネットワークを使って、ニューラルネットワークの出力を一つの値から条件付き分布へ広げます。"),
            ("main", "普通の回帰では、入力エックスが決まると、ターゲットの平均を一つ返すと考えがちです。"),
            ("main", "しかし逆問題では、同じ観測に対して、正しい答えが二つ以上あることがあります。"),
            ("main", "そのとき平均だけを返すと、どの正解にも近くない中間の値を選んでしまいます。"),
            ("summary", "混合密度ネットワークは、一つの予測値ではなく、あり得る答えの分布を出すための方法です。"),
        ],
    },
    {
        "id": "scene02",
        "title": "条件付き混合分布",
        "lines": [
            ("main", "目標は、入力エックスを条件にしたターゲットの分布、ピー、ティー、バー、エックスを表すことです。"),
            ("main", "PRML では、ガウス分布を複数混ぜた形を使います。"),
            ("main", "各成分には、重みパイ、中心ミュー、幅シグマがあります。"),
            ("main", "重要なのは、この三つが固定値ではなく、入力エックスの関数になることです。"),
            ("summary", "エックスごとに混合比、中心、幅を変えれば、単峰にも多峰にもなる条件付き分布を表せます。"),
        ],
    },
    {
        "id": "scene03",
        "title": "ネットワーク出力から分布パラメータへ",
        "lines": [
            ("main", "混合密度ネットワークでは、普通のニューラルネットワークが混合分布のパラメータを出力します。"),
            ("main", "混合係数はゼロ以上で、全部足すと一になる必要があるので、ソフトマックスで変換します。"),
            ("main", "シグマは正でなければならないので、指数関数で変換します。"),
            ("main", "ミューは実数をそのまま取れるので、出力活性を直接使えます。"),
            ("summary", "ネットワークの生の出力を、確率分布として意味を持つ値へ変換するのがポイントです。"),
        ],
    },
    {
        "id": "scene04",
        "title": "尤度で学習する",
        "lines": [
            ("main", "学習では、観測されたターゲットに高い密度を与えるように、重みを調整します。"),
            ("main", "二乗誤差ではなく、混合分布の負の対数尤度を最小化します。"),
            ("main", "ある点がどの成分から出たらしそうかは、ガンマ、つまり事後的な責務として計算できます。"),
            ("main", "この責務を使うと、混合係数、中心、幅に対する誤差信号を通常の逆伝播で流せます。"),
            ("summary", "MDN の学習は、分布全体の尤度を上げるように、成分ごとの責務を使って進みます。"),
        ],
    },
    {
        "id": "scene05",
        "title": "入力で変わる密度",
        "lines": [
            ("main", "学習後のモデルを、条件付き密度の地図として見てみます。"),
            ("main", "横軸が入力エックス、縦軸がターゲットティーで、明るい場所ほど確率密度が高い場所です。"),
            ("main", "左端や右端では一つの山だけになり、中央では三つの山が同時に現れます。"),
            ("main", "ネットワーク出力は連続な関数ですが、混合比を変えることで、密度の形は多峰になります。"),
            ("summary", "MDN は、各エックスで必要なだけの候補を、密度として残せます。"),
        ],
    },
    {
        "id": "scene06",
        "title": "平均とモード",
        "lines": [
            ("main", "条件付き分布が手に入ると、そこから必要な要約量を選べます。"),
            ("main", "条件付き平均は、成分の中心を混合係数で重み付き平均したものです。"),
            ("main", "ただし多峰分布では、平均が低密度の谷に落ちることがあります。"),
            ("main", "実際の制御や選択では、一番重みの大きい成分の中心、つまり近似的なモードを使うほうが自然な場合があります。"),
            ("summary", "MDN は平均だけでなく、モードや分散など、用途に合った予測を後から選べます。"),
        ],
    },
    {
        "id": "scene07",
        "title": "まとめ",
        "lines": [
            ("main", "混合密度ネットワークは、ニューラルネットワークと混合モデルを組み合わせた条件付き密度モデルです。"),
            ("main", "ネットワークは、入力からパイ、ミュー、シグマを出力し、ソフトマックスと指数関数で制約を満たします。"),
            ("main", "負の対数尤度を最小化すれば、通常の逆伝播で学習できます。"),
            ("main", "逆問題のように答えが複数ある場面では、平均の一点予測よりも、分布を持つ予測のほうが情報を失いません。"),
            ("summary", "五章六節の核心は、回帰を一つの点から、条件付き確率密度の予測へ拡張することです。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 5.6 narration WAV files with VOICEVOX.")
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
