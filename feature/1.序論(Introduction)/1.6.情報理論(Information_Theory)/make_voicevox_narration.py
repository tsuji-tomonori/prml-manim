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
        "title": "情報量は驚きの大きさ",
        "lines": [
            ("main", "情報理論では、情報を、驚きの大きさとして測ります。"),
            ("main", "ほぼ確実な出来事を聞いても、新しく分かることは少ない。"),
            ("main", "めったに起きない出来事を聞くと、たくさんの情報を受け取ります。"),
            ("main", "ピーアールエムエルでは、この情報量をマイナスログ確率で表します。"),
            ("summary", "確率が小さいほど、情報量は大きくなります。"),
        ],
    },
    {
        "id": "scene02",
        "title": "エントロピーは平均情報量",
        "lines": [
            ("main", "一回ごとの情報量を、確率で平均したものがエントロピーです。"),
            ("main", "八通りがすべて同じ確率なら、どれが出たかを伝えるには三ビット必要です。"),
            ("main", "でも、よく出る値とめったに出ない値が分かっているなら、よく出る値に短い符号を割り当てられます。"),
            ("main", "このとき平均の符号長は短くなります。"),
            ("summary", "エントロピーは、最短の平均符号長の下限として読めます。"),
        ],
    },
    {
        "id": "scene03",
        "title": "広がった分布ほどエントロピーが高い",
        "lines": [
            ("main", "分布の形を見ると、エントロピーの直感がつかみやすくなります。"),
            ("main", "確率が一部の値に集中していると、結果は予想しやすく、エントロピーは小さくなります。"),
            ("main", "確率が多くの値へ広がると、どれが出るか分かりにくくなり、エントロピーは大きくなります。"),
            ("main", "離散分布では、同じ個数の状態なら一様分布でエントロピーが最大になります。"),
        ],
    },
    {
        "id": "scene04",
        "title": "連続分布とガウス分布",
        "lines": [
            ("main", "連続値では、エントロピーは積分で定義され、微分エントロピーと呼ばれます。"),
            ("main", "同じ平均と分散だけを固定したとき、もっともエントロピーが大きい分布はガウス分布です。"),
            ("main", "分散が大きくなるほど、分布は広がり、不確かさも増えます。"),
            ("summary", "ガウス分布は、平均と分散だけを知る状況で、余計な仮定を足さない分布として現れます。"),
        ],
    },
    {
        "id": "scene05",
        "title": "条件付きエントロピー",
        "lines": [
            ("main", "二つの変数エックスとワイを同時に考えます。"),
            ("main", "もしエックスを先に知っているなら、ワイを説明するために必要な情報は減るかもしれません。"),
            ("main", "この残りの不確かさが、条件付きエントロピーです。"),
            ("main", "同時に説明する情報量は、エックスを説明する情報量と、エックスを知ったあとにワイを説明する情報量に分解できます。"),
        ],
    },
    {
        "id": "scene06",
        "title": "KLダイバージェンス",
        "lines": [
            ("main", "次に、真の分布ピーを、近似分布キューで代用する状況を考えます。"),
            ("main", "キューを使って符号化すると、ピーを知っている場合より平均的に余計な情報が必要になります。"),
            ("main", "この余分な情報量が、相対エントロピー、またはケーエルダイバージェンスです。"),
            ("question", "これは普通の距離と同じですか？"),
            ("main", "いいえ。向きを入れ替えると値が変わるので、対称な距離ではありません。"),
            ("summary", "ただし、ピーとキューが完全に一致するときだけゼロになります。"),
        ],
    },
    {
        "id": "scene07",
        "title": "KL最小化と最尤推定",
        "lines": [
            ("main", "機械学習では、真の分布ピーは分かりません。"),
            ("main", "代わりに、ピーから来たと考える訓練データだけを観測します。"),
            ("main", "ケーエルダイバージェンスをデータで近似すると、パラメータに依存する部分は、マイナス対数尤度になります。"),
            ("main", "つまり、ケーエルダイバージェンスを小さくすることは、最尤推定とつながります。"),
        ],
    },
    {
        "id": "scene08",
        "title": "相互情報量",
        "lines": [
            ("main", "最後に、二つの変数がどれくらい関係しているかを測ります。"),
            ("main", "もしエックスとワイが独立なら、同時分布は周辺分布の積に分解できます。"),
            ("main", "そこからのずれをケーエルダイバージェンスで測ったものが、相互情報量です。"),
            ("summary", "相互情報量は、ワイを知ることでエックスの不確かさがどれだけ減るか、として読めます。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 1.6 narration WAV files with VOICEVOX.")
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
