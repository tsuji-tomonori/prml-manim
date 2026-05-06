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

PAUSE_SECONDS = 0.3

SCENES = [
    {
        "id": "scene01",
        "title": "事後確率を直接学ぶ",
        "lines": [
            ("main", "四章の前半では、クラスごとの密度を仮定し、ベイズの定理で事後確率を作りました。"),
            ("main", "これは生成モデルの考え方です。"),
            ("main", "一方で分類に必要なのは、最終的には各クラスの事後確率です。"),
            ("focus", "そこで確率的識別モデルでは、条件付き確率ピー、シー、ギブン、エックスを直接モデル化します。"),
            ("main", "ガウス密度や事前確率を別々に合わせる代わりに、ロジスティックやソフトマックスの形を置き、そのパラメータを直接最尤推定します。"),
            ("summary", "特に特徴数が大きいと、密度全体を作るよりも調整するパラメータが少なくなる利点があります。"),
        ],
    },
    {
        "id": "scene02",
        "title": "固定基底関数で境界を曲げる",
        "lines": [
            ("main", "ここからは、入力エックスをそのまま使う代わりに、固定された基底関数ファイ、エックスへ変換してから分類します。"),
            ("main", "モデルは特徴空間では線形です。"),
            ("main", "けれども、特徴そのものがエックスの非線形変換なので、元の入力空間では曲がった境界になります。"),
            ("main", "これは回帰の線形基底関数モデルと同じ発想です。"),
            ("focus", "ただし、固定基底はクラスの重なりを魔法のように消すものではありません。"),
            ("summary", "重なりがあるときは、確率そのものをうまく表すことが大事になります。"),
        ],
    },
    {
        "id": "scene03",
        "title": "ロジスティック回帰は分類確率のモデル",
        "lines": [
            ("main", "二クラス分類では、クラス一の事後確率をシグモイドで表します。"),
            ("main", "線形スコア、エー、イコール、ダブリュー転置ファイを計算し、それをゼロから一の確率へ押し込みます。"),
            ("main", "シグモイドが二分の一になる場所、つまりエーがゼロの場所が決定境界です。"),
            ("focus", "名前はロジスティック回帰ですが、ここで予測しているのは連続値ではなくクラスの確率です。"),
            ("summary", "その確率を決定理論に渡せば、単にラベルを返すだけでなく、迷いの大きさも扱えます。"),
        ],
    },
    {
        "id": "scene04",
        "title": "交差エントロピーと単純な勾配",
        "lines": [
            ("main", "目標値は、クラス一なら一、クラス二ならゼロです。"),
            ("main", "各点の確率をベルヌーイ分布として掛け合わせ、マイナス対数を取ると、交差エントロピー誤差になります。"),
            ("focus", "この誤差をダブリューで微分すると、とても見通しのよい形になります。"),
            ("main", "各点の寄与は、予測ワイから目標ティーを引いた誤差に、特徴ベクトルファイを掛けたものです。"),
            ("main", "ただし、データが完全に線形分離できると、最尤推定では重みの大きさが無限に伸び、シグモイドが階段関数のように急になります。"),
            ("summary", "この特異性は、事前分布や正則化で抑える必要があります。"),
        ],
    },
    {
        "id": "scene05",
        "title": "IRLSは重み付き最小二乗を繰り返す",
        "lines": [
            ("main", "線形回帰なら、二乗誤差は二次関数なので、一回の最小二乗で解けました。"),
            ("main", "ロジスティック回帰ではシグモイドが入るため、閉じた形の解はありません。"),
            ("main", "そこでニュートン法を使い、現在のダブリューの近くで誤差を二次関数として近似します。"),
            ("focus", "ヘッセ行列には、アール、イコール、ワイかける一マイナスワイ、という重みが出てきます。"),
            ("main", "この重みは、確率が半々に近い境界付近で大きく、ほぼゼロか一の点では小さくなります。"),
            ("summary", "つまりアイアールエルエスは、不確かな点を重く見ながら、重み付き最小二乗を繰り返す方法です。"),
        ],
    },
    {
        "id": "scene06",
        "title": "多クラスではsoftmaxを使う",
        "lines": [
            ("main", "クラスが三つ以上あるときは、シグモイドの代わりにソフトマックスを使います。"),
            ("main", "各クラスに線形スコア、エー、ケーを作り、その指数を全クラスの指数の和で割ります。"),
            ("main", "こうすると、すべての確率はゼロから一の間に入り、合計は一になります。"),
            ("main", "目標は、正しいクラスだけが一で、ほかはゼロのワンオブケー符号です。"),
            ("summary", "負の対数尤度は多クラス交差エントロピーになり、勾配はここでも、予測マイナス目標、かける特徴、という同じ形になります。"),
        ],
    },
    {
        "id": "scene07",
        "title": "probitはしきい値ノイズのCDF",
        "lines": [
            ("main", "シグモイド以外の活性化関数も考えられます。"),
            ("main", "ひとつの見方は、線形スコアがランダムなしきい値シータを超えたらクラス一、という noisy threshold モデルです。"),
            ("main", "しきい値に分布を置くと、クラス一になる確率は、その分布の累積分布関数になります。"),
            ("focus", "しきい値ノイズを標準ガウスにすれば、活性化関数はプロビット関数です。"),
            ("main", "多くの場合、ロジスティック回帰とプロビット回帰の結果は似ています。"),
            ("main", "ただし尾の減り方が違うため、間違った側に遠く離れた外れ値への感度は変わります。"),
            ("summary", "ラベル誤りを明示的に入れるなら、確率イプシロンで目標が反転するモデルとして扱えます。"),
        ],
    },
    {
        "id": "scene08",
        "title": "canonical linkが勾配をそろえる",
        "lines": [
            ("main", "最後に、なぜ同じ勾配の形が何度も出てきたのかを整理します。"),
            ("main", "線形回帰では、予測と目標の差に特徴を掛ける形でした。"),
            ("main", "ロジスティック回帰でも、ソフトマックスでも、対応する交差エントロピーを使うと同じ形になります。"),
            ("focus", "ピーアールエムエルはこれを、指数型分布族と canonical link の組み合わせとして説明します。"),
            ("main", "平均を線形スコアへ戻すリンク関数を、分布に対応する自然な形に選ぶと、余分な微分項が打ち消されます。"),
            ("summary", "確率を直接学ぶ識別モデルは、この単純な勾配と、事後確率を返せる実用性の両方を持っています。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 4.3 narration WAV files with VOICEVOX.")
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
