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
    "emphasis": {
        "label": "VOICEVOX:WhiteCUL",
        "id": 23,
        "speed_scale": 1.05,
        "intonation_scale": 0.9,
    },
    "summary": {
        "label": "VOICEVOX:WhiteCUL",
        "id": 23,
        "speed_scale": 1.03,
        "intonation_scale": 0.88,
    },
}

PAUSE_SECONDS = 0.3

SCENES = [
    {
        "id": "scene01",
        "title": "固定基底から学習される基底へ",
        "lines": [
            ("main", "これまでの線形モデルでは、入力をまず固定された基底関数に通し、その出力を重みつきで足し合わせていました。"),
            ("main", "この形は扱いやすい一方で、どんな基底関数を置くかを人が先に決める必要があります。"),
            ("main", "ニューラルネットワークでは、この基底関数そのものにもパラメータを持たせます。"),
            ("main", "つまり、出力側の重みだけでなく、入力をどう特徴に変えるかも、訓練データから調整します。"),
            ("summary", "この節では、その関数がどのように入力から出力まで組み立てられるかを見ます。"),
        ],
    },
    {
        "id": "scene02",
        "title": "順伝播でネットワーク関数を評価する",
        "lines": [
            ("main", "まず入力を重みつきで足し合わせ、隠れユニットの活性エー・ジェーを作ります。"),
            ("main", "その活性を、シグモイドやタンエイチのような微分可能な非線形関数エイチに通すと、隠れユニットの出力ゼット・ジェーになります。"),
            ("main", "次に、ゼット・ジェーをもう一度重みつきで足し合わせ、出力ユニットの活性エー・ケーを作ります。"),
            ("main", "最後に、問題に合った出力活性化を通してワイ・ケーを得ます。"),
            ("summary", "この左から右への評価が、順伝播、つまりフォワード・プロパゲーションです。"),
        ],
    },
    {
        "id": "scene03",
        "title": "隠れユニットは学習される非線形基底",
        "lines": [
            ("main", "隠れユニットを一つ取り出すと、入力の線形結合に非線形関数をかけた、小さな曲線として見られます。"),
            ("main", "重みとバイアスを変えると、この曲線の位置や向き、立ち上がり方が変わります。"),
            ("main", "出力層では、それらの隠れユニットをさらに重みつきで足し合わせます。"),
            ("main", "そのためネットワーク全体は、固定された基底を足すのではなく、データに合わせて動く基底を組み合わせる関数になります。"),
            ("summary", "複数の隠れユニットが協調することで、一つ一つは単純でも複雑な形を作れます。"),
        ],
    },
    {
        "id": "scene04",
        "title": "出力活性化はタスクで変わる",
        "lines": [
            ("main", "出力ユニットの活性化は、何を予測したいかで変わります。"),
            ("main", "回帰なら、出力を実数として使いたいので、恒等写像を使います。"),
            ("main", "二値分類なら、各出力をゼロから一の範囲に入れるため、ロジスティックシグモイドを使います。"),
            ("main", "多クラス分類なら、全クラスの確率が足して一になるように、ソフトマックスを使います。"),
            ("summary", "この選び方は、次の五・二節で、尤度や誤差関数と結びついて説明されます。"),
        ],
    },
    {
        "id": "scene05",
        "title": "非線形がなければ多層でも線形",
        "lines": [
            ("main", "もし隠れユニットの活性化関数がすべて線形なら、何層に重ねても結果は一つの線形変換にまとめられます。"),
            ("main", "行列を何度か掛けても、全体としては別の一つの行列になるからです。"),
            ("main", "その場合、隠れ層を増やしても、本質的な表現力はあまり増えません。"),
            ("main", "一方で、途中にタンエイチのような非線形を入れると、線形変換だけでは作れない曲がった関数を表せます。"),
            ("summary", "ニューラルネットワークの表現力は、この線形変換と非線形変換の組み合わせから生まれます。"),
        ],
    },
    {
        "id": "scene06",
        "title": "Feed-forward topology",
        "lines": [
            ("main", "ネットワーク図は、より一般的な接続にも拡張できます。"),
            ("main", "入力から出力へ直接つなぐスキップ・コネクションを入れてもよいですし、すべての接続を持たないスパースな構造にしてもかまいません。"),
            ("main", "重要なのは、矢印をたどって閉じたサイクルがないことです。"),
            ("main", "閉路がなければ、入力が与えられたとき、前のユニットから順番に値を計算でき、出力は入力の決定的な関数になります。"),
            ("summary", "この意味で、ここで扱う図は確率的グラフィカルモデルではなく、関数を表す計算グラフです。"),
        ],
    },
    {
        "id": "scene07",
        "title": "Universal approximator の意味",
        "lines": [
            ("main", "十分な数の隠れユニットを使えば、フィードフォワードネットワークはとても広い範囲の関数を近似できます。"),
            ("main", "ピーアールエムエルでは、二層ネットワークが、二乗、サイン、絶対値、ステップ関数のような形を近似する例を示しています。"),
            ("main", "この性質は、ニューラルネットワークがユニバーサル・アプロキシメーターと呼ばれる理由です。"),
            ("emphasis", "ただし、近似できるパラメータが存在することと、訓練データからそのパラメータをうまく見つけられることは別です。"),
            ("summary", "次の節以降では、まさにその学習方法が中心になります。"),
        ],
    },
    {
        "id": "scene08",
        "title": "重み空間の対称性",
        "lines": [
            ("main", "最後に、同じ入出力関数を表す重みが一つとは限らない、という性質を見ておきます。"),
            ("main", "タンエイチは符号を反転すると出力の符号も反転します。"),
            ("main", "そこで、ある隠れユニットに入る重みを反転し、その隠れユニットから出る重みも反転すると、最終的な足し合わせは同じになります。"),
            ("main", "また、隠れユニットの順番を入れ替えても、足し合わせる項の順番が変わるだけなので、関数は変わりません。"),
            ("summary", "この重み空間の対称性は、多くの場合は目立ちませんが、ベイズ的なモデル比較では後で重要になります。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 5.1 narration WAV files with VOICEVOX.")
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
