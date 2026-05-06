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
        "title": "生成モデルで分類を見る",
        "lines": [
            ("main", "四章二節では、分類を確率モデルとして見直します。"),
            ("main", "識別関数では、入力エックスを見て、すぐにクラスを分ける面を置きました。"),
            ("main", "ここではまず、各クラスがどんなデータを生成しやすいか、ピーエックス、バー、シーケーをモデル化します。"),
            ("main", "さらにクラスの事前確率、ピーシーケーも合わせます。"),
            ("summary", "最後にベイズの定理で、入力エックスが来たときの事後確率ピーシーケー、バー、エックスを計算します。"),
        ],
    },
    {
        "id": "scene02",
        "title": "二クラスではロジットがシグモイドに入る",
        "lines": [
            ("main", "二クラスの場合、クラス一の事後確率は、尤度と事前確率の積を正規化したものです。"),
            ("main", "この式は、クラス一とクラス二の比の対数、つまりログオッズを使うと、ロジスティックシグモイドの形に書き換えられます。"),
            ("main", "エーが大きいほど、クラス一を支持する証拠が強いことを意味します。"),
            ("main", "エーがゼロなら両クラスは同じだけ支持され、事後確率は一対二、つまりゼロ点五です。"),
            ("summary", "重要なのは、エーが単純な関数になるかどうかです。"),
        ],
    },
    {
        "id": "scene03",
        "title": "共有共分散ガウスから線形境界が出る",
        "lines": [
            ("main", "連続入力の例として、各クラスの条件付き密度をガウス分布とします。"),
            ("main", "すべてのクラスが同じ共分散行列シグマを共有していると仮定します。"),
            ("main", "ガウス分布の指数部には、エックスの二次項が含まれます。"),
            ("main", "しかし二つのクラスで共分散が同じなら、比を取ると二次項が打ち消されます。"),
            ("summary", "その結果、シグモイドの中身は、ダブリュー転置エックス足すダブリューゼロという線形関数になります。"),
        ],
    },
    {
        "id": "scene04",
        "title": "事前確率は境界を平行移動する",
        "lines": [
            ("main", "事前確率ピーシーケーは、データを見る前に、どのクラスがどれくらい出やすいかを表します。"),
            ("main", "共有共分散ガウスの場合、事前確率はバイアス項ダブリューゼロにだけ入ります。"),
            ("main", "クラス一の事前確率を大きくすると、クラス一と判定される領域が広がります。"),
            ("main", "逆に小さくすると、クラス一にはより強い証拠が必要になります。"),
            ("summary", "重みベクトルの向きは変わらないので、境界は平行に動きます。"),
        ],
    },
    {
        "id": "scene05",
        "title": "多クラスではソフトマックスになる",
        "lines": [
            ("main", "クラスが三つ以上ある場合も考え方は同じです。"),
            ("main", "各クラスについて、ピーエックス、バー、シーケーとピーシーケーの積を作り、全クラスで正規化します。"),
            ("main", "対数を取ったスコア、エーケーが線形なら、事後確率はソフトマックスで表されます。"),
            ("main", "共有共分散のガウス分布では、ここでも二次項が打ち消され、エーケーは線形関数になります。"),
            ("summary", "最も大きい事後確率どうしが等しいところが、線形の決定境界です。"),
        ],
    },
    {
        "id": "scene06",
        "title": "最尤推定で部品を学習する",
        "lines": [
            ("main", "生成モデルでは、まずクラスごとの密度と事前確率をデータから推定します。"),
            ("main", "二クラスの共有共分散ガウスなら、クラス一の事前確率パイは、クラス一のデータ数を全データ数で割った値です。"),
            ("main", "平均ミュー一とミュー二は、それぞれのクラスに属する入力ベクトルの平均です。"),
            ("main", "共有共分散シグマは、クラスごとの共分散を、データ数で重み付けして平均します。"),
            ("summary", "ただしガウス分布の最尤推定なので、外れ値には強くありません。"),
        ],
    },
    {
        "id": "scene07",
        "title": "離散特徴ではナイーブベイズが線形スコアを作る",
        "lines": [
            ("main", "入力がゼロか一の離散特徴でできている場合を考えます。"),
            ("main", "特徴がディー個あると、一般の表は二のディー乗個の状態を持ち、すぐに大きくなります。"),
            ("main", "そこで、クラスが決まれば各特徴は条件付き独立だと仮定します。"),
            ("main", "このナイーブベイズの仮定では、クラス条件付き分布は特徴ごとのベルヌーイ分布の積になります。"),
            ("summary", "対数を取ると、スコアは各エックスアイの線形結合として表されます。"),
        ],
    },
    {
        "id": "scene08",
        "title": "生成モデルから識別モデルへの橋渡し",
        "lines": [
            ("main", "この節の結論は、かなり広い生成モデルから、線形スコアのシグモイドやソフトマックスが自然に出てくる、ということです。"),
            ("main", "共有共分散のガウス、ナイーブベイズ、そしてスケールを共有した指数型分布族は、その代表例です。"),
            ("main", "一方で、クラスごとに別の共分散を許すと、二次項は打ち消されず、境界は二次曲線になります。"),
            ("main", "また、密度の仮定がデータに合わないと、事後確率も悪くなります。"),
            ("summary", "次の確率的識別モデルでは、密度を別々に作らず、事後確率の形を直接学習します。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 4.2 narration WAV files with VOICEVOX.")
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
        manifest["scenes"].append(generate_scene(args.base_url, scene))

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total_duration = sum(float(scene["duration"]) for scene in manifest["scenes"])
    print(f"Wrote {manifest_path}")
    print(f"Total narration duration: {total_duration:.3f} sec")


if __name__ == "__main__":
    main()
