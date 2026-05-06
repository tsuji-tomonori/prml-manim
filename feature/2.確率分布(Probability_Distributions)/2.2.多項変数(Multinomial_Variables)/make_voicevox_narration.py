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
        "title": "二値から多値へ",
        "lines": [
            ("main", "前の節では、表か裏、ゼロか一、という二値変数を見ました。"),
            ("main", "でも実際の分類では、候補が二つだけとは限りません。"),
            ("main", "手書き数字なら、ゼロから九までの十通り。"),
            ("main", "文書分類なら、ニュース、スポーツ、経済、天気、という複数の状態があります。"),
            ("summary", "ピーアールエムエル二点二節では、このような多項変数を扱います。"),
        ],
    },
    {
        "id": "scene02",
        "title": "1-of-K 表現",
        "lines": [
            ("main", "多項変数は、ケー個の状態のうち、ちょうど一つだけが選ばれる変数です。"),
            ("main", "ピーアールエムエルでは、これをワンオブケー表現で書きます。"),
            ("main", "たとえば六通りのうち三番目が出たなら、ベクトルは、ゼロ、ゼロ、一、ゼロ、ゼロ、ゼロです。"),
            ("main", "一が立っている場所だけが、観測された状態を表します。"),
            ("summary", "全ての成分を足すと、必ず一になります。"),
        ],
    },
    {
        "id": "scene03",
        "title": "カテゴリ分布",
        "lines": [
            ("main", "それぞれの状態には、起こりやすさを表すミューケーを置きます。"),
            ("main", "ミューは確率なので、全てゼロ以上で、合計は一です。"),
            ("main", "ワンオブケー表現を使うと、一回の観測の確率を、積の形で短く書けます。"),
            ("main", "観測された場所だけ指数が一になり、ほかはゼロになります。"),
            ("summary", "つまり式は長く見えても、実際には選ばれたミューを一つ取り出しているだけです。"),
        ],
    },
    {
        "id": "scene04",
        "title": "データはカウントに縮約される",
        "lines": [
            ("main", "独立な観測が何回もあると、尤度は各観測の確率を掛け合わせたものになります。"),
            ("main", "しかし、全部の順番を覚えておく必要はありません。"),
            ("main", "この分布では、各状態が何回出たか、というカウントだけで尤度が決まります。"),
            ("main", "このカウントをエムケーと書きます。"),
            ("summary", "エムケーは、この分布にとって十分統計量です。"),
        ],
    },
    {
        "id": "scene05",
        "title": "最尤推定は相対頻度",
        "lines": [
            ("main", "では、ミューをデータから推定するとどうなるでしょうか。"),
            ("main", "ミューの合計が一、という制約の下で尤度を最大にします。"),
            ("main", "結果はとても直感的です。"),
            ("main", "各ミューの最尤推定値は、その状態が出た回数を、全体の回数で割ったものになります。"),
            ("summary", "つまり最尤推定では、確率を観測された相対頻度として読むことになります。"),
        ],
    },
    {
        "id": "scene06",
        "title": "多項分布",
        "lines": [
            ("main", "今度は、順番ではなく、カウントの組そのものの確率を考えます。"),
            ("main", "同じカウントでも、観測の並び方はたくさんあります。"),
            ("main", "その並び方の数を掛けたものが、多項分布です。"),
            ("main", "係数は、エヌ個の観測を、ケー個のグループへ分ける方法の数を表します。"),
            ("summary", "二値の二項分布を、ケー通りに広げたものだと見れば十分です。"),
        ],
    },
    {
        "id": "scene07",
        "title": "ディリクレ分布",
        "lines": [
            ("main", "ベイズ的に考えるなら、ミューそのものにも事前分布を置きます。"),
            ("main", "ミューは合計が一なので、自由に動ける場所は単体と呼ばれる領域の上だけです。"),
            ("main", "多項分布に対する便利な事前分布が、ディリクレ分布です。"),
            ("main", "パラメータのアルファケーは、その状態をどれくらい先に信じているかを表します。"),
            ("summary", "アルファは、仮想的な事前カウントとして解釈できます。"),
        ],
    },
    {
        "id": "scene08",
        "title": "事前カウントに実データを足す",
        "lines": [
            ("main", "ディリクレ事前分布に多項分布のデータを掛けると、事後分布もディリクレ分布になります。"),
            ("main", "変わるのはパラメータだけです。"),
            ("main", "事前のアルファに、観測されたカウントエムを足します。"),
            ("main", "これは、前の節のベータ分布とベルヌーイ分布で見た関係の、多値版です。"),
            ("summary", "次の節では、連続値を扱う中心的な分布、ガウス分布へ進みます。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 2.2 narration WAV files with VOICEVOX.")
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
