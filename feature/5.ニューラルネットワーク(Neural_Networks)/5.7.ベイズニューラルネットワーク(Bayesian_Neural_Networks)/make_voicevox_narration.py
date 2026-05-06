from __future__ import annotations

import argparse
import http.client
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
        "title": "点推定から分布へ",
        "lines": [
            ("main", "ここまでのニューラルネットワークでは、学習後に一組の重みを選びました。"),
            ("main", "その重みを固定すれば、入力から出力への関数が一本決まります。"),
            ("main", "でも、データが少ない場所では、同じくらいありそうな重みが他にも残ります。"),
            ("main", "ベイズニューラルネットワークでは、重みを一つに決め打ちしません。"),
            ("summary", "重みそのものに確率分布を持たせ、予測の不確かさまで表します。"),
        ],
    },
    {
        "id": "scene02",
        "title": "重みの事前分布",
        "lines": [
            ("main", "学習前には、重みについての事前分布を置きます。"),
            ("main", "たとえば、重みはゼロの近くにありやすく、大きすぎる値は起きにくい、と考えます。"),
            ("main", "これは、正則化で大きすぎる係数を抑えた考え方とつながっています。"),
            ("main", "ただし今回は、罰則の値ではなく、どの重みがどれくらいありそうか、という分布として扱います。"),
            ("main", "画面の曲線は、事前分布から重みをいくつか引いたときにできる関数です。"),
        ],
    },
    {
        "id": "scene03",
        "title": "尤度で重みをふるいにかける",
        "lines": [
            ("main", "データを観測すると、重みの候補は同じ扱いではなくなります。"),
            ("main", "青い点をよく説明する重みは、尤度が高くなります。"),
            ("main", "点から大きく外れる重みは、尤度が低くなります。"),
            ("question", "つまり、候補の曲線を全部残すけれど、重み付けを変えるのですか？"),
            ("main", "その通りです。ベイズでは、良い候補だけを一つ選ぶのではなく、候補全体の重みを更新します。"),
        ],
    },
    {
        "id": "scene04",
        "title": "事後分布",
        "lines": [
            ("main", "更新後の重みの分布を、事後分布と呼びます。"),
            ("main", "式で書けば、事後分布は、尤度と事前分布を掛け、エビデンスで正規化したものです。"),
            ("main", "エビデンスは、分布全体の面積を一にそろえる役割を持ちます。"),
            ("main", "データが増えるほど、ありそうな重みの範囲は狭くなります。"),
            ("summary", "ただし、完全に一点になるわけではありません。不確かさは事後分布の幅として残ります。"),
        ],
    },
    {
        "id": "scene05",
        "title": "予測分布",
        "lines": [
            ("main", "新しい入力で予測するときは、重みを一つだけ使いません。"),
            ("main", "事後分布の中の重みを全部考え、その重みごとの予測を平均します。"),
            ("main", "これが予測分布です。"),
            ("main", "データが集まっている範囲では、候補の曲線はよくそろいます。"),
            ("main", "一方、データがない外側では、曲線が広がり、不確かさが大きく見えます。"),
            ("summary", "ベイズニューラルネットワークの出力は、値だけでなく、どれくらい自信があるかを含みます。"),
        ],
    },
    {
        "id": "scene06",
        "title": "ラプラス近似",
        "lines": [
            ("main", "問題は、ニューラルネットワークの事後分布を正確に扱うのが難しいことです。"),
            ("main", "そこでピーアールエムエルでは、近似の入口としてラプラス近似を使います。"),
            ("main", "まず、事後分布が最も高い重み、つまりマップ解を探します。"),
            ("main", "その周りを、二次関数の形で近似します。"),
            ("main", "二次の曲がり具合はヘッセ行列で決まり、対応するガウス分布の幅を決めます。"),
        ],
    },
    {
        "id": "scene07",
        "title": "エビデンスと複雑さ",
        "lines": [
            ("main", "ベイズの枠組みでは、モデルの複雑さも自動的に効いてきます。"),
            ("main", "とても複雑なネットワークは、たくさんの関数を表せます。"),
            ("main", "しかし、データをよく説明する重みの領域が狭ければ、全体としてのエビデンスは伸びにくくなります。"),
            ("main", "これは、よく当たることと、広い範囲で自然に当たることを同時に見る考え方です。"),
            ("summary", "ベイズのエビデンスは、当てはまりと複雑さのバランスを一つの量で評価します。"),
        ],
    },
    {
        "id": "scene08",
        "title": "まとめ",
        "lines": [
            ("main", "ベイズニューラルネットワークは、ニューラルネットワークに確率論を重ねたものです。"),
            ("main", "学習とは、重みの分布を、事前分布から事後分布へ更新することです。"),
            ("main", "予測では、事後分布の重みを平均し、不確かさも一緒に出します。"),
            ("main", "正確な計算は難しいため、ラプラス近似などの近似推論が重要になります。"),
            ("summary", "一本の答えではなく、ありそうな答えの分布を見る。これが、この節の中心です。"),
        ],
    },
]


def post_json(base_url: str, path: str, params: dict[str, str | int], body: bytes | None = None) -> bytes:
    url = f"{base_url.rstrip('/')}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(1, 4):
        request = urllib.request.Request(url, data=body, method="POST")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (TimeoutError, OSError, http.client.HTTPException, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(1.5 * attempt)
    raise RuntimeError("unreachable retry loop")


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
    frame_chunks: list[bytes] = []
    for line_number, (speaker_key, text) in enumerate(scene["lines"], start=1):  # type: ignore[index]
        print(f"{scene_id} line {line_number}", flush=True)
        wav_bytes = synthesize(base_url, speaker_key, text)
        tmp_line_path = OUTPUT_DIR / "_line.wav"
        tmp_line_path.write_bytes(wav_bytes)
        try:
            with wave.open(str(tmp_line_path), "rb") as source:
                current_params = (source.getnchannels(), source.getsampwidth(), source.getframerate())
                if params is None:
                    params = current_params
                elif current_params != params:
                    raise RuntimeError(f"Unexpected WAV params: {current_params}, expected {params}")
                frame_chunks.append(source.readframes(source.getnframes()))
        finally:
            tmp_line_path.unlink(missing_ok=True)
        assert params is not None
        frame_chunks.append(b"\x00" * int(params[2] * PAUSE_SECONDS) * params[0] * params[1])
        time.sleep(0.03)

    if params is None:
        raise RuntimeError(f"No WAV frames generated for {scene_id}")
    with wave.open(str(tmp_output_path), "wb") as output:
        output.setnchannels(params[0])
        output.setsampwidth(params[1])
        output.setframerate(params[2])
        for frames in frame_chunks:
            output.writeframes(frames)
    tmp_output_path.replace(output_path)
    return {
        "id": scene_id,
        "title": scene["title"],
        "path": str(output_path.relative_to(SCENE_DIR)),
        "duration": round(wav_duration(output_path), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PRML 5.7 narration WAV files with VOICEVOX.")
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
