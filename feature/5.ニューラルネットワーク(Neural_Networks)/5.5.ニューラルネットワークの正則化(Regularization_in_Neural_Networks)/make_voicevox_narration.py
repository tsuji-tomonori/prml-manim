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

PAUSE_SECONDS = 0.3

SCENES = [
    {
        "id": "scene01",
        "title": "隠れユニット数だけで複雑さは決まらない",
        "lines": [
            ("main", "ニューラルネットワークでは、入力ユニット数と出力ユニット数はデータとタスクでほぼ決まります。"),
            ("main", "一方で、隠れユニット数エムは設計者が選ぶ自由なパラメータです。"),
            ("main", "エムが小さすぎると単純すぎて当てはまらず、大きすぎると訓練データの細かな揺れまで覚えやすくなります。"),
            ("main", "ただし、非線形ネットワークでは局所最小もあるため、汎化誤差はエムだけの単純な関数にはなりません。"),
            ("summary", "そこで、少し大きめのネットワークを用意し、正則化で実効的な複雑さを制御します。"),
        ],
    },
    {
        "id": "scene02",
        "title": "重み減衰は大きすぎる重みを抑える",
        "lines": [
            ("main", "もっとも基本的な正則化は、二乗和の重み減衰です。"),
            ("main", "元の誤差に、重みベクトルの二乗ノルムを足します。"),
            ("main", "ラムダが小さいと、訓練データへ強く合わせるため、曲線は細かく暴れます。"),
            ("main", "ラムダを大きくすると、大きな重みが抑えられ、関数の暴れも小さくなります。"),
            ("summary", "ラムダは、データへの適合と、滑らかさの釣り合いを決めるつまみです。"),
        ],
    },
    {
        "id": "scene03",
        "title": "ガウス事前分布として見る",
        "lines": [
            ("main", "重み減衰は、重みに平均ゼロのガウス事前分布を置いたときの負の対数としても読めます。"),
            ("main", "ただし、すべての重みとバイアスを同じ強さで縮める単純な事前分布には限界があります。"),
            ("main", "入力のスケールや出力のスケールが変わると、同じ関数を表すために必要な重みの大きさも変わるからです。"),
            ("main", "PRML では、層ごとの重みとバイアスに別々の精度を持たせる、より整合的なガウス事前分布を導入します。"),
            ("summary", "正則化係数は一つとは限らず、どの重み群をどれだけ抑えるかを分けて考えられます。"),
        ],
    },
    {
        "id": "scene04",
        "title": "早期終了は暗黙の正則化",
        "lines": [
            ("main", "早期終了は、明示的な正則化項を足さずに複雑さを制御する方法です。"),
            ("main", "訓練誤差は反復とともに下がり続けても、検証誤差は途中から上がり始めることがあります。"),
            ("main", "検証誤差が最小になる時点で学習を止めると、訓練データの細部を覚え込む前のモデルを選べます。"),
            ("main", "二次近似の重み空間では、原点から少し進んだ場所で止めることが、重み減衰で縮めた解に似た働きをします。"),
            ("summary", "学習をどこで止めるかも、実効的なモデル複雑さを決める正則化です。"),
        ],
    },
    {
        "id": "scene05",
        "title": "不変性を学習に入れる",
        "lines": [
            ("main", "画像認識では、同じ数字が少し横にずれても、同じクラスとして扱いたい。"),
            ("main", "このように、入力変換に対して出力が変わらない性質を不変性と呼びます。"),
            ("main", "大量の変換例を訓練データに入れれば、ネットワークは不変性を近似的に学べます。"),
            ("main", "しかし、変換の種類が増えると組み合わせが急増し、必要なデータも計算も大きくなります。"),
            ("summary", "そのため、データ拡張、正則化項、特徴抽出、ネットワーク構造の四つの方向で不変性を組み込みます。"),
        ],
    },
    {
        "id": "scene06",
        "title": "タンジェント伝播",
        "lines": [
            ("main", "タンジェント伝播は、不変性を正則化項として入れる方法です。"),
            ("main", "入力エックスを少し回転や平行移動すると、入力空間の中で小さな曲線を描きます。"),
            ("main", "その曲線の接線方向をタウとします。"),
            ("main", "ネットワークの出力が、この接線方向にほとんど変わらなければ、局所的にその変換へ不変です。"),
            ("summary", "そこで、ヤコビアンとタウの積が小さくなるような正則化項を加えます。"),
        ],
    },
    {
        "id": "scene07",
        "title": "変換データで訓練することとの関係",
        "lines": [
            ("main", "変換したデータをたくさん作って訓練する方法と、タンジェント伝播は密接に関係します。"),
            ("main", "変換量が平均ゼロで小さいとき、変換後データでの平均二乗誤差を展開できます。"),
            ("main", "すると、元の誤差に、入力方向の微分を抑える項が足された形になります。"),
            ("main", "特に入力に小さなランダムノイズを加える場合は、ティホノフ正則化と呼ばれる形につながります。"),
            ("summary", "データ拡張は、関数を入力の小さな変化に鈍感にする正則化として理解できます。"),
        ],
    },
    {
        "id": "scene08",
        "title": "畳み込みネットワークは構造で不変性を入れる",
        "lines": [
            ("main", "不変性をネットワーク構造に組み込む代表例が、畳み込みネットワークです。"),
            ("main", "画像では近いピクセルほど関係が強いため、各ユニットは小さな局所領域だけを見ます。"),
            ("main", "同じ重みを画像上の別の場所でも共有すれば、同じ特徴を別の位置で検出できます。"),
            ("main", "さらにサブサンプリングで細かな位置ずれへの感度を下げます。"),
            ("summary", "局所受容野、重み共有、サブサンプリングが、パラメータ数を抑えつつ平行移動への頑健さを作ります。"),
        ],
    },
    {
        "id": "scene09",
        "title": "ソフト重み共有とまとめ",
        "lines": [
            ("main", "最後に、ソフト重み共有を見ます。"),
            ("main", "畳み込みのように完全に同じ重みに固定するのではなく、重みがいくつかのグループの中心に近づくよう正則化します。"),
            ("main", "これは、重みの事前分布をガウス混合として置くことに対応します。"),
            ("main", "学習では、重みだけでなく、各グループの中心や広がりも同時に調整します。"),
            ("summary", "五点五節の正則化は、重みを小さくするだけでなく、止める時刻、不変性、構造、重みのまとまりまで含む広い考え方です。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 5.5 narration WAV files with VOICEVOX.")
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
