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
    "formula": {
        "label": "VOICEVOX:WhiteCUL",
        "id": 23,
        "speed_scale": 1.04,
        "intonation_scale": 0.92,
    },
    "summary": {
        "label": "VOICEVOX:WhiteCUL",
        "id": 23,
        "speed_scale": 1.03,
        "intonation_scale": 0.9,
    },
}

PAUSE_SECONDS = 0.3

SCENES = [
    {
        "id": "scene01",
        "title": "重みを一点ではなく分布として扱う",
        "lines": [
            ("main", "ロジスティック回帰では、重みベクトル w を一つ選ぶと、シグモイド関数でクラス確率が決まります。"),
            ("main", "ベイズロジスティック回帰では、この w を一点推定ではなく、事後分布として持ちます。"),
            ("main", "すると、分類境界だけでなく、その境界をどれくらい信じてよいかも表せます。"),
            ("main", "ただし、事後分布は、事前分布と、データ点ごとのシグモイド尤度の積になります。"),
            ("summary", "この積を正確に正規化することも、予測時に積分することも、解析的には難しくなります。"),
        ],
    },
    {
        "id": "scene02",
        "title": "事前分布と尤度から事後分布を作る",
        "lines": [
            ("main", "まず、重み w にガウス事前分布を置きます。"),
            ("main", "データを観測すると、それぞれの点が、シグモイド尤度として w の値に制約をかけます。"),
            ("formula", "ログ事後分布は、事前分布の二次項と、交差エントロピー型の対数尤度の和になります。"),
            ("main", "この形は、線形回帰のようなガウス同士の共役な積ではありません。"),
            ("summary", "そこで、事後分布そのものを扱う代わりに、近くでガウスに置き換える準備をします。"),
        ],
    },
    {
        "id": "scene03",
        "title": "ラプラス近似でMAPの近くをガウスにする",
        "lines": [
            ("main", "ラプラス近似では、まず事後分布が最大になる点、つまり MAP 解を探します。"),
            ("main", "次に、その点での曲率をヘッセ行列で測ります。"),
            ("formula", "PRML 4.5.1 では、近似事後分布を、平均 w MAP、共分散 S N のガウス分布として書きます。"),
            ("main", "曲率が大きい方向は、少し動かすだけで事後確率が下がるので、分散が小さくなります。"),
            ("summary", "曲率が小さい方向は、データだけでは重みをよく決められていない方向です。"),
        ],
    },
    {
        "id": "scene04",
        "title": "MAPのロジスティック分類器",
        "lines": [
            ("main", "二次元のデータで、MAP 解のロジスティック回帰を見てみます。"),
            ("main", "背景の色は、クラス一である確率です。赤に近いほどクラス一、青に近いほどクラス二です。"),
            ("main", "白い線が、確率がちょうど 0.5 になる決定境界です。"),
            ("main", "MAP だけを見ると、重みは一つなので、各場所の確率も一つの値に決まります。"),
            ("summary", "しかし、データが少ない領域では、この確率をそのまま信じすぎるのは危険です。"),
        ],
    },
    {
        "id": "scene05",
        "title": "重みの不確実性はロジットの不確実性になる",
        "lines": [
            ("main", "新しい入力 phi に対して、ロジスティック回帰は a イコール w 転置 phi という一次元のロジットを作ります。"),
            ("main", "w の事後分布をガウスで近似しているので、a もまたガウス分布になります。"),
            ("formula", "その平均は w MAP 転置 phi、分散は phi 転置 S N phi です。"),
            ("main", "データの近くでは、ロジットの分布は狭くなります。"),
            ("main", "データから離れるほど、同じ MAP 予測でも、ロジットの分布は広くなります。"),
            ("summary", "予測では、このロジットの分布全体に対してシグモイドを平均します。"),
        ],
    },
    {
        "id": "scene06",
        "title": "予測分布は確率を0.5側へ戻す",
        "lines": [
            ("main", "予測したいクラス確率は、シグモイドに事後分布を掛けて、重み w について積分したものです。"),
            ("main", "ラプラス近似のもとでは、これは、シグモイドと一次元ガウスの積分になります。"),
            ("formula", "PRML 4.5.2 では、この積分を、カッパを掛けた平均ロジットのシグモイドで近似します。"),
            ("main", "分散が大きいほどカッパは小さくなり、確率は 0.5 に近づきます。"),
            ("main", "一方で、確率 0.5 の境界は、平均ロジットがゼロの場所なので、MAP の境界と同じです。"),
            ("summary", "変わるのは境界そのものではなく、境界から離れた確率の自信の持ち方です。"),
        ],
    },
    {
        "id": "scene07",
        "title": "4.5節の要点",
        "lines": [
            ("main", "ベイズロジスティック回帰では、重みを事後分布として扱うことで、分類の不確実性を予測に反映します。"),
            ("main", "正確な事後分布と予測分布は扱いにくいため、PRML 4.5 ではラプラス近似を使います。"),
            ("main", "MAP を平均にし、ヘッセ行列の逆から共分散を作り、ガウス近似を得ます。"),
            ("main", "その上で、ロジット a の一次元ガウスに落とし込み、シグモイドとの積分を近似します。"),
            ("summary", "この考え方は、ガウス過程分類や変分ロジスティック回帰にもつながる基本パターンです。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 4.5 narration WAV files with VOICEVOX.")
    parser.add_argument("--base-url", default="http://127.0.0.1:50021", help="VOICEVOX Engine URL")
    parser.add_argument("--from-scene", default="scene01", help="First scene id to regenerate")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_index = next((i for i, scene in enumerate(SCENES) if scene["id"] == args.from_scene), 0)
    manifest: dict[str, object] = {
        "speakers": {key: {"label": value["label"], "id": value["id"]} for key, value in SPEAKERS.items()},
        "scenes": [],
    }
    scenes: list[dict[str, object]] = manifest["scenes"]  # type: ignore[assignment]
    for scene in SCENES[:start_index]:
        output_path = OUTPUT_DIR / f"{scene['id']}.wav"
        scenes.append(
            {
                "id": scene["id"],
                "title": scene["title"],
                "path": str(output_path.relative_to(SCENE_DIR)),
                "duration": round(wav_duration(output_path), 3),
            }
        )
    for scene in SCENES[start_index:]:
        result = generate_scene(args.base_url, scene)
        scenes.append(result)
        print(f"{result['id']} {result['duration']}s {result['path']}", flush=True)

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest {manifest_path.relative_to(SCENE_DIR)}", flush=True)


if __name__ == "__main__":
    main()
