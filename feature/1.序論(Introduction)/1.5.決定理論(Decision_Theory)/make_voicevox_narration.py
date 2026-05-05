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

PAUSE_SECONDS = 0.32

SCENES = [
    {
        "id": "scene01",
        "title": "確率から行動へ",
        "lines": [
            ("main", "確率論は、不確かさを数字で表すための道具でした。"),
            ("main", "でも、実際の問題では、確率を見たあとに、何かを決める必要があります。"),
            ("main", "画像を見て、病気の可能性を出すだけでなく、治療するか、追加検査に回すかを選ぶ。"),
            ("main", "この、確率から具体的な行動を選ぶ部分が、決定理論です。"),
            ("summary", "推論は確率を求める段階。決定は、その確率を使って行動を選ぶ段階です。"),
        ],
    },
    {
        "id": "scene02",
        "title": "事後確率と決定境界",
        "lines": [
            ("main", "まず、間違いの回数だけを小さくしたい場合を考えます。"),
            ("main", "入力エックスが来たら、それぞれのクラスの事後確率を比べます。"),
            ("main", "二つのクラスなら、事後確率が大きいほうを選ぶのが最適です。"),
            ("main", "画面の縦線は、クラスを切り替える決定境界です。"),
            ("main", "境界を動かすと、間違いになる面積が変わります。"),
            ("summary", "交点で切ると、各エックスでより確からしいクラスを選ぶことになります。"),
        ],
    },
    {
        "id": "scene03",
        "title": "損失行列と期待損失",
        "lines": [
            ("main", "ただし、間違いの数だけを見ればよいとは限りません。"),
            ("main", "医療診断では、健康な人を追加検査に回す間違いと、病気を見逃す間違いの重さは違います。"),
            ("main", "そこで、判断ごとの痛みを、損失行列として書きます。"),
            ("main", "本当のクラスが分からないので、事後確率で重みづけした平均の損失を比べます。"),
            ("main", "病気の確率が小さく見えても、見逃しの損失が大きいなら、治療側の判断が選ばれます。"),
            ("summary", "最適な判断とは、期待損失を最小にする判断です。"),
        ],
    },
    {
        "id": "scene04",
        "title": "棄却オプション",
        "lines": [
            ("main", "事後確率がどちらも決め手に欠ける場所では、無理に分類しない選択もあります。"),
            ("main", "これを棄却オプションと呼びます。"),
            ("main", "しきい値シータを決め、最大の事後確率がそれ以下なら、自動判断を保留します。"),
            ("main", "医療の例なら、あいまいな画像は専門家に回す、という使い方です。"),
            ("summary", "棄却は逃げではなく、難しいケースの損失を管理するための決定です。"),
        ],
    },
    {
        "id": "scene05",
        "title": "推論と決定の三つの作り方",
        "lines": [
            ("main", "分類器の作り方には、大きく三つの段階があります。"),
            ("main", "一つ目は、各クラスで入力がどう分布するかをモデル化し、ベイズの定理で事後確率を出す方法です。"),
            ("main", "これは生成モデルと呼ばれ、外れ値検出やデータ生成にも使えます。"),
            ("main", "二つ目は、事後確率そのものを直接モデル化する識別モデルです。"),
            ("main", "三つ目は、確率を出さず、入力からラベルへ直接写す識別関数です。"),
            ("summary", "後ろへ行くほど単純ですが、損失変更や棄却には事後確率があるほうが柔軟です。"),
        ],
    },
    {
        "id": "scene06",
        "title": "回帰の決定理論",
        "lines": [
            ("main", "回帰でも、確率から一つの予測値を決める必要があります。"),
            ("main", "入力エックスを固定したとき、ターゲットティーには条件付き分布があります。"),
            ("main", "二乗損失を使うなら、最適な予測は、この分布の平均です。"),
            ("main", "式で書くと、ワイエックスは、ティーの条件付き期待値になります。"),
            ("main", "平均から離れた値を選ぶほど、二乗された外し具合の平均が大きくなります。"),
            ("summary", "最小二乗の背後には、二乗損失と条件付き平均という決定理論があります。"),
        ],
    },
    {
        "id": "scene07",
        "title": "次は情報理論へ",
        "lines": [
            ("main", "決定理論は、確率だけではなく、行動の損失まで含めて考えます。"),
            ("main", "同じ事後確率でも、損失行列が変われば、最適な行動は変わります。"),
            ("main", "分類では、事後確率、損失、棄却が判断を作ります。"),
            ("main", "回帰では、損失の選び方が、平均、中央値、最頻値といった予測の意味を変えます。"),
            ("summary", "次の一・六節では、不確かさを情報量として測る、情報理論に進みます。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 1.5 narration WAV files with VOICEVOX.")
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
