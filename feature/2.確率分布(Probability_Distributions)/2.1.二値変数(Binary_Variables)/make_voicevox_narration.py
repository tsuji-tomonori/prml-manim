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
        "title": "二値変数とは",
        "lines": [
            ("main", "今回は、ピーアールエムエル二点一、二値変数を見ます。"),
            ("main", "二値変数とは、結果が二つだけの変数です。"),
            ("main", "コインなら表か裏。検査なら陽性か陰性。クリックなら、した、しない。"),
            ("main", "ここでは、結果を一なら成功、ゼロなら失敗、と書きます。"),
            ("main", "たった二つの値ですが、この章の確率分布の基本形がここから始まります。"),
        ],
    },
    {
        "id": "scene02",
        "title": "Bernoulli 分布",
        "lines": [
            ("main", "二値変数エックスに対して、一が出る確率をミューと置きます。"),
            ("main", "すると、ゼロが出る確率は、一マイナスミューです。"),
            ("main", "この二つを一つの式にまとめたものが、ベルヌーイ分布です。"),
            ("main", "エックスが一なら、式はミューになります。"),
            ("main", "エックスがゼロなら、式は一マイナスミューになります。"),
            ("main", "画面のつまみでミューを動かすと、二本の棒の高さが反対向きに変わります。"),
        ],
    },
    {
        "id": "scene03",
        "title": "最尤推定",
        "lines": [
            ("main", "次に、データからミューを推定します。"),
            ("main", "十回観測して、一が七回、ゼロが三回出たとします。"),
            ("main", "このデータが出やすくなるミューを選ぶのが、最尤推定です。"),
            ("main", "ベルヌーイ分布では、尤度はミューの七乗かける、一マイナスミューの三乗になります。"),
            ("main", "曲線の山は、七割のところで最大になります。"),
            ("summary", "つまり最尤推定では、ミューは一の割合、エム割るエヌになります。"),
        ],
    },
    {
        "id": "scene04",
        "title": "少ないデータの不安定さ",
        "lines": [
            ("main", "ただし、最尤推定だけを見ると、データが少ないときに強く言い切りすぎることがあります。"),
            ("main", "三回中三回が一なら、最尤推定はミューイコール一です。"),
            ("question", "でも、たまたま三回続いただけかもしれません。"),
            ("main", "ここで、ミューそのものにも確率分布を置きます。"),
            ("main", "二値変数のミューに対して便利なのが、ベータ分布です。"),
            ("main", "ベータ分布は、ミューがゼロから一の間のどこにありそうかを表します。"),
        ],
    },
    {
        "id": "scene05",
        "title": "Beta 事前分布から事後分布へ",
        "lines": [
            ("main", "ベータ分布には、エーとビーという二つのパラメータがあります。"),
            ("main", "直感的には、エーは一を見た分の重み、ビーはゼロを見た分の重みです。"),
            ("main", "観測で一がエム回、ゼロがエル回出ると、エーにはエムを足し、ビーにはエルを足します。"),
            ("main", "これがベータ分布とベルヌーイ分布のきれいな関係です。"),
            ("main", "同じ形の分布のまま、パラメータだけが更新されます。"),
            ("summary", "この性質を、共役事前分布と呼びます。"),
        ],
    },
    {
        "id": "scene06",
        "title": "逐次更新",
        "lines": [
            ("main", "更新は、データをまとめて見なくても、一つずつ進められます。"),
            ("main", "一を見たらエーを一つ増やす。ゼロを見たらビーを一つ増やす。"),
            ("main", "曲線は、観測が増えるほど狭くなり、ミューのありそうな場所へ集まっていきます。"),
            ("main", "最尤推定は観測された割合だけを見ます。"),
            ("main", "ベイズ推定では、事前の重みと観測データを足し合わせて、分布全体を更新します。"),
        ],
    },
    {
        "id": "scene07",
        "title": "予測分布と次節への橋渡し",
        "lines": [
            ("main", "最後に、次の一回が一になる確率を考えます。"),
            ("main", "ベイズ的な予測では、ミューの一点だけではなく、事後分布全体を平均します。"),
            ("main", "結果は、エムプラスエーを、エヌプラスエー足すビーで割った形になります。"),
            ("main", "これは、観測された一の数に、事前分布の重みを足した割合です。"),
            ("main", "二値変数では、一かゼロの二択でした。"),
            ("main", "次の二点二では、三択以上の多項変数へ広げます。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 2.1 narration WAV files with VOICEVOX.")
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
