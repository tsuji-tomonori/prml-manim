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
    "focus": {
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

PAUSE_SECONDS = 0.28

SCENES = [
    {
        "id": "scene01",
        "title": "逆伝播は学習法ではなく微分の計算手順",
        "lines": [
            ("main", "この節の目的は、フィードフォワードネットワークの誤差関数を、すべての重みで効率よく微分することです。"),
            ("main", "バックプロパゲーションという言葉は、学習アルゴリズム全体を指して使われることもあります。"),
            ("focus", "ここでは、重みをどう更新するかではなく、勾配をどう計算するかに限定して考えます。"),
            ("main", "入力から出力へ値を流し、そのあと誤差の情報を出力から入力側へ戻します。"),
            ("summary", "前向きと後ろ向きの局所的なメッセージパッシングが、誤差逆伝播です。"),
        ],
    },
    {
        "id": "scene02",
        "title": "重みの微分は局所情報の積になる",
        "lines": [
            ("main", "まず一つの重みに注目します。重みダブリュー、ジェーアイは、入力側の値ズィーアイを受け取り、ユニットジェーの総入力エージェーにだけ直接影響します。"),
            ("main", "したがって、連鎖律を使うと、誤差の重み微分は、誤差のエージェー微分と、エージェーの重み微分の積になります。"),
            ("focus", "デルタジェーを、誤差をエージェーで微分した量、と定義します。"),
            ("main", "また、エージェーを重みダブリュー、ジェーアイで微分すると、入力側の値ズィーアイが残ります。"),
            ("summary", "つまり、重みの勾配は、出力側のデルタと入力側の値の積、デルタジェーかけるズィーアイです。"),
        ],
    },
    {
        "id": "scene03",
        "title": "前向き計算で必要な値を保存する",
        "lines": [
            ("main", "勾配を計算する前に、まず通常の前向き計算を行います。"),
            ("main", "各ユニットでは、前の層から来た値を重み付きで足し合わせて、総入力エーを作ります。"),
            ("main", "次に、微分可能な非線形関数エイチを通して、出力ズィーを作ります。"),
            ("focus", "後ろ向き計算では、このエーとズィーが必要になります。"),
            ("summary", "前向き計算は、予測値を出すだけでなく、微分に使う中間値を準備する段階でもあります。"),
        ],
    },
    {
        "id": "scene04",
        "title": "出力ユニットのデルタから始める",
        "lines": [
            ("main", "後ろ向き計算は出力ユニットから始まります。"),
            ("main", "二乗和誤差と線形出力、または正準リンクと対応する誤差関数を使う場合、出力のデルタは単純になります。"),
            ("focus", "デルタケーは、予測値ワイケーから目標値ティーケーを引いたものです。"),
            ("main", "これは、出力側でどちら向きにどれだけずれているかを表す誤差信号です。"),
            ("summary", "出力デルタが分かると、それを前の層へ順番に戻せます。"),
        ],
    },
    {
        "id": "scene05",
        "title": "隠れユニットのデルタを逆向きに集める",
        "lines": [
            ("main", "隠れユニットは目標値と直接比べられません。そこで、そのユニットが接続している先のユニットからデルタを受け取ります。"),
            ("main", "ユニットジェーから先へ出る重みと、先のデルタを掛けて足し合わせます。"),
            ("main", "さらに、ジェー自身の活性化関数の傾き、エイチプライム、エージェーを掛けます。"),
            ("focus", "これが、デルタジェー イコール エイチプライム、エージェー、かける、重みとデルタの総和です。"),
            ("summary", "誤差の情報は、接続を逆向きにたどりながら、各ユニットの局所微分で調整されます。"),
        ],
    },
    {
        "id": "scene06",
        "title": "二層ネットワークで勾配を作る",
        "lines": [
            ("main", "具体例として、隠れ層にハイパボリックタンジェント、出力層に線形出力を持つ二層ネットワークを考えます。"),
            ("main", "前向き計算では、入力から隠れ層のエーとズィーを作り、そこから出力ワイを作ります。"),
            ("main", "出力デルタは、ワイ引くティーです。隠れデルタは、一マイナス、ズィーの二乗を掛けて、次層の重み付きデルタを戻します。"),
            ("focus", "最後に各重みの勾配は、デルタと入力側の値の外積として並びます。"),
            ("summary", "全重みの勾配が、局所的な掛け算だけで一度に得られます。"),
        ],
    },
    {
        "id": "scene07",
        "title": "効率と実装確認",
        "lines": [
            ("main", "一回の前向き計算は、重み数をダブリューとすると、およそオーダーダブリューの計算です。"),
            ("main", "有限差分で全重みを一つずつ揺らすと、各重みに前向き計算が必要になり、全体はオーダーダブリュー二乗になります。"),
            ("focus", "逆伝播は、一回の前向き計算と一回の後ろ向き計算で、全重みの勾配をまとめて求めるため、オーダーダブリューで済みます。"),
            ("main", "ただし実装確認には、少数の重みで中心差分と比べるのが有効です。"),
            ("summary", "実際の学習では逆伝播で勾配を計算し、その勾配を別の最適化法が利用します。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 5.3 narration WAV files with VOICEVOX.")
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
