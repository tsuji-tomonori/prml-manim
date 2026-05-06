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
        "speed_scale": 1.03,
        "intonation_scale": 0.9,
    },
    "summary": {
        "label": "VOICEVOX:WhiteCUL",
        "id": 23,
        "speed_scale": 1.05,
        "intonation_scale": 0.9,
    },
}

PAUSE_SECONDS = 0.3

SCENES = [
    {
        "id": "scene01",
        "title": "識別関数は直接クラスを選ぶ",
        "lines": [
            ("main", "第4章は、分類のための線形モデルを扱います。入力ベクトル x を、離散的なクラス C k のどれかへ割り当てる問題です。"),
            ("main", "その中で4.1節の識別関数は、確率をいったん作らず、入力から直接クラスを選びます。"),
            ("main", "複数クラスなら、クラスごとにスコア y k x を計算し、一番大きいスコアのクラスを選ぶ、という見方ができます。"),
            ("main", "これは、事後確率 p C k given x を推定する方法や、生成モデル p x given C k からベイズの定理を使う方法とは違う入口です。"),
            ("summary", "まずは、決定境界が線形になる識別関数を、幾何で理解します。"),
        ],
    },
    {
        "id": "scene02",
        "title": "二クラス線形識別関数の幾何",
        "lines": [
            ("main", "二クラスの場合、もっとも単純な線形識別関数は、y x イコール w 転置 x プラス w ゼロです。"),
            ("main", "y x がゼロ以上ならクラス1、ゼロ未満ならクラス2に割り当てます。"),
            ("main", "境界は y x イコールゼロなので、二次元では直線、高次元では超平面になります。"),
            ("main", "重みベクトル w は、この境界に垂直です。つまり w が境界の向きを決めます。"),
            ("main", "一方、バイアス w ゼロは、境界を原点からどれだけずらすかを決めます。"),
            ("summary", "さらに y x を w の長さで割ると、境界からの符号付き距離になります。"),
        ],
    },
    {
        "id": "scene03",
        "title": "多クラスではスコアを同時に比較する",
        "lines": [
            ("main", "多クラス分類を、二クラス識別器の寄せ集めで作ることも考えられます。"),
            ("main", "しかし one versus rest や one versus one を単純に組み合わせると、どのクラスとも言い切れない、あいまいな領域が生じることがあります。"),
            ("main", "PRML 4.1では、この問題を避けるため、K 個の線形関数 y k x をまとめて使います。"),
            ("main", "入力 x は、y k x が他のすべての y j x より大きいとき、クラス C k に割り当てられます。"),
            ("main", "二つのクラスの境界は y k x イコール y j x で表されるので、やはり線形境界です。"),
            ("summary", "この形式では、それぞれの決定領域は凸な領域になります。"),
        ],
    },
    {
        "id": "scene04",
        "title": "最小二乗を分類に使うと何が起きるか",
        "lines": [
            ("main", "次に、識別関数のパラメータをどう学習するかを考えます。最初の候補は、回帰で使った最小二乗です。"),
            ("main", "1 of K の target ベクトルに対して、予測 W 転置 x チルダ が target に近づくように二乗誤差を最小にします。"),
            ("main", "この方法は、擬似逆行列を使って W イコール X チルダ dagger T という閉形式解を持ちます。"),
            ("main", "ただし、分類では問題があります。出力の和が1になっても、各成分が0から1の範囲に収まるとは限りません。"),
            ("main", "さらに二乗誤差は、正しい側に大きく離れた点まで強く罰するため、外れ値に境界が引っぱられます。"),
            ("summary", "閉形式で速いことと、分類に適した損失であることは、別の問題です。"),
        ],
    },
    {
        "id": "scene05",
        "title": "Fisher の線形識別",
        "lines": [
            ("main", "Fisher の線形識別は、二クラスのデータを一つの軸 y イコール w 転置 x に射影する方法として考えます。"),
            ("main", "単にクラス平均を結ぶ向きへ射影すると、クラス内のばらつきが大きい方向を選んでしまい、射影後に重なりが残ることがあります。"),
            ("main", "Fisher の考え方は、射影後の平均の差を大きくしつつ、各クラス内のばらつきを小さくすることです。"),
            ("main", "この比が Fisher criterion で、分子はクラス間の分散、分母はクラス内の分散を表します。"),
            ("summary", "最適な方向は、S W の逆行列に、クラス平均の差を掛けた向きになります。"),
        ],
    },
    {
        "id": "scene06",
        "title": "Fisher は方向、判定にはしきい値",
        "lines": [
            ("main", "注意したいのは、Fisher が直接クラス名を返す関数ではなく、よい射影方向を与える方法だという点です。"),
            ("main", "射影後の一次元の値 y に、しきい値 y ゼロを置くことで、はじめて二クラスの判定になります。"),
            ("main", "PRML ではさらに、二クラスの場合、特別な target coding を使った最小二乗が Fisher と同じ向きを与えることを示します。"),
            ("main", "また K クラスへ拡張すると、Fisher で得られる有効な線形特徴の数は、最大でも K マイナス1本です。"),
            ("summary", "Fisher は、分類前にクラスを分けやすい低次元表現を作る方法として見ると分かりやすくなります。"),
        ],
    },
    {
        "id": "scene07",
        "title": "パーセプトロン更新",
        "lines": [
            ("main", "最後の例は、Rosenblatt のパーセプトロンです。入力を特徴ベクトル phi x に変換し、w 転置 phi x の符号で二クラスを判定します。"),
            ("main", "target は、クラス1をプラス1、クラス2をマイナス1とするのが便利です。"),
            ("main", "正しく分類された点では、重みを変えません。間違えた点だけについて、w に phi x n かける t n を足します。"),
            ("main", "この更新は、間違えた点を正しい側へ押し戻すように、決定境界を動かします。"),
            ("main", "データが線形分離可能なら、有限回で正しく分類する解に到達することが保証されます。"),
            ("summary", "ただし、線形分離できないデータでは、パーセプトロンは収束しません。"),
        ],
    },
    {
        "id": "scene08",
        "title": "識別関数のまとめ",
        "lines": [
            ("main", "4.1節の識別関数は、入力からクラスへの直接写像を作る、非確率的な分類方法です。"),
            ("main", "最小二乗は閉形式で簡単ですが、分類の損失としては外れ値や確率解釈に弱さがあります。"),
            ("main", "Fisher は、クラス間を離し、クラス内を縮める射影方向を与えます。"),
            ("main", "パーセプトロンは、誤分類だけを使ってオンラインに重みを更新しますが、線形分離不能な場合の限界があります。"),
            ("summary", "次の節では、p x given C k や p C k given x を扱う、確率的な線形分類へ進みます。"),
        ],
    },
]


def post_json(base_url: str, path: str, params: dict[str, str | int], body: bytes | None = None) -> bytes:
    url = f"{base_url.rstrip('/')}{path}?{urllib.parse.urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(1, 5):
        request = urllib.request.Request(url, data=body, method="POST")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - transient VOICEVOX HTTP disconnects are retried here.
            last_error = error
            if attempt == 4:
                break
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"VOICEVOX request failed after retries: {path}") from last_error


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
    parser = argparse.ArgumentParser(description="Generate PRML 4.1 narration WAV files with VOICEVOX.")
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
