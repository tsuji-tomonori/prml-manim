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

PAUSE_SECONDS = 0.34

SCENES = [
    {
        "id": "scene01",
        "title": "不確かさを数で扱う",
        "lines": [
            ("main", "前回は、データの点から一本の曲線を選びました。"),
            ("main", "でも実際のデータには、ノイズがあります。"),
            ("main", "同じ入力エックスでも、観測されるティーは少しずつ揺れます。"),
            ("main", "だから知りたいのは、答えが一つだけ、という話ではありません。"),
            ("main", "どの値がどれくらいありそうかを、まとめて表す必要があります。"),
            ("summary", "そのための言葉が、確率です。"),
        ],
    },
    {
        "id": "scene02",
        "title": "確率は可能性の重み",
        "lines": [
            ("main", "確率は、起こり方に付ける重みだと思えます。"),
            ("main", "重みはゼロ以上で、全部を足すと一になります。"),
            ("main", "この棒グラフでは、どのクラスがどれくらいありそうかを表しています。"),
            ("main", "一番高い棒だけを見るのではなく、ほかの可能性もまだ残しておきます。"),
            ("main", "機械学習では、この残り方がとても大事です。"),
        ],
    },
    {
        "id": "scene03",
        "title": "同時確率と周辺化",
        "lines": [
            ("main", "二つの変数を同時に見るときは、表の一マスが一つの組み合わせを表します。"),
            ("main", "たとえば、クラスと特徴が同時にこうなる、という重みです。"),
            ("main", "行を全部足すと、そのクラス全体の確率になります。"),
            ("main", "列を全部足すと、その特徴全体の確率になります。"),
            ("summary", "見たい変数だけを残して、ほかを足し消す。これが周辺化です。"),
        ],
    },
    {
        "id": "scene04",
        "title": "条件付き確率と積の法則",
        "lines": [
            ("main", "条件付き確率は、条件を聞いたあとの確率です。"),
            ("main", "エックスが分かったあとで、クラスがどれくらいありそうかを見る、という形です。"),
            ("main", "同時確率は、条件付き確率と、条件そのものの確率に分けられます。"),
            ("main", "式では、ピーエックスカンマワイは、ピーエックスかけるピーワイ条件エックス、と書けます。"),
            ("main", "全体の重みを、入口の重みと、そこから先の割合に分けているだけです。"),
        ],
    },
    {
        "id": "scene05",
        "title": "ベイズの定理",
        "lines": [
            ("main", "ベイズの定理は、見方を反対向きにするための道具です。"),
            ("main", "原因からデータが出る確率が分かっているとします。"),
            ("main", "でも実際に欲しいのは、データを見たあとで、どの原因がありそうかです。"),
            ("main", "事前の重みと、データの出やすさを掛け合わせます。"),
            ("main", "最後に、全部を足して一になるように割り直します。"),
            ("summary", "これが、観測したあとに信念を更新する、ということです。"),
        ],
    },
    {
        "id": "scene06",
        "title": "密度は面積で読む",
        "lines": [
            ("main", "連続的な値では、一点ちょうどの確率ではなく、区間の確率を考えます。"),
            ("main", "曲線の高さは確率そのものではありません。"),
            ("main", "区間の下にある面積が、その範囲に入る確率です。"),
            ("main", "面積を全部足すと一になるように作られた曲線を、確率密度と呼びます。"),
            ("main", "ガウス分布は、平均の近くが高く、遠くへ行くほど低くなる密度です。"),
        ],
    },
    {
        "id": "scene07",
        "title": "期待値と分散",
        "lines": [
            ("main", "分布を一つの代表値で見るなら、期待値を使います。"),
            ("main", "期待値は、確率の重みを付けた平均です。"),
            ("main", "棒が重い場所に、平均の印が引き寄せられます。"),
            ("main", "一方で、平均だけでは広がり方は分かりません。"),
            ("main", "平均からどれくらい散らばっているかを見る量が、分散です。"),
            ("summary", "中心と広がりを分けて見ると、不確かさを扱いやすくなります。"),
        ],
    },
    {
        "id": "scene08",
        "title": "回帰を確率モデルにする",
        "lines": [
            ("main", "確率論を使うと、前回の曲線フィッティングも見方が変わります。"),
            ("main", "予測曲線は、ティーの平均を表している、と考えます。"),
            ("main", "実際の観測値は、その平均のまわりにノイズを持って散らばります。"),
            ("main", "式では、ティーはワイエックスダブリューを中心にしたガウス分布から出る、と書けます。"),
            ("main", "一本の曲線だけでなく、どれくらい不確かかも一緒に持てるようになります。"),
            ("summary", "ここから、尤度、ベイズ推論、確率的なモデル選択へ進んでいきます。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 1.2 narration WAV files with VOICEVOX.")
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
