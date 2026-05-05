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
        "title": "オープニング",
        "lines": [
            ("main", "データは、ただの数字の集まりに見えます。"),
            ("main", "でも、その中に何度も現れる形があるとき、私たちはそれをパターンと呼びます。"),
            ("main", "機械学習の最初の仕事は、この形を見つけることです。"),
            ("main", "ただし、見つけたいのは、過去のデータそのものではありません。"),
            ("main", "まだ見ていないデータにも通用する、背後の規則性です。"),
        ],
    },
    {
        "id": "scene02",
        "title": "手書き数字から数値ベクトルへ",
        "lines": [
            ("main", "人間には、これは数字の、さん、に見えます。"),
            ("main", "でもコンピュータに渡すとき、それはピクセルごとの明るさの数字です。"),
            ("main", "ピーアールエムエルの例では、画像サイズが二十八かける二十八です。"),
            ("main", "さらに白黒画像なので、一つのピクセルには、明るさを表す数字が一つ入ります。"),
            ("main", "つまりこの例の画像一枚は、七百八十四個の数字のリストだと思えば十分です。"),
            ("main", "その数字の並びから、これは、さん、っぽい、と判断する。"),
            ("main", "これがパターン認識のひとつの例です。"),
        ],
    },
    {
        "id": "scene03",
        "title": "学習とは generalization",
        "lines": [
            ("main", "機械学習で本当にやりたいことは、過去のデータを覚えることではありません。"),
            ("main", "練習問題を使って、初見問題に強くなることです。"),
            ("main", "ピーアールエムエルでは、この能力を、ジェネラリゼーション、つまり汎化と呼びます。"),
            ("main", "訓練データに合うだけでは足りません。"),
            ("main", "まだ見ていない入力に対して、うまく予測できることが目的です。"),
        ],
    },
    {
        "id": "scene04",
        "title": "今回は regression",
        "lines": [
            ("main", "学習問題には、いくつかの種類があります。"),
            ("main", "画像が、ゼロから九のどれかを当てるなら、分類問題です。"),
            ("main", "一方で、入力エックスから、連続的な数字ティーを予測するなら、回帰問題です。"),
            ("main", "回帰では、エックスを入れるとティーを返す関数を作りたい。"),
            ("main", "その関数の候補として、今回は多項式を使います。"),
            ("main", "つまり多項式曲線フィッティングは、回帰を一番小さく観察するための実験です。"),
        ],
    },
    {
        "id": "scene05",
        "title": "10個の点から見えない曲線を想像する",
        "lines": [
            ("main", "ここに十個の点があります。"),
            ("main", "横軸は入力エックス、縦軸は予測したい値ティー。"),
            ("main", "この点たちは、完全にバラバラではなさそうです。"),
            ("main", "何か、波のような形に沿っている気がします。"),
            ("main", "でも、学習アルゴリズムに見えているのは青い点だけです。"),
            ("main", "実は今回は、裏側にこの緑の曲線がある、という設定でデータを作っています。"),
            ("main", "ただし、学習する側にはこの曲線は見えていません。"),
            ("main", "問題は、有限個の点から、まだ見ていない場所を予測できるかです。"),
        ],
    },
    {
        "id": "scene06",
        "title": "多項式モデル",
        "lines": [
            ("main", "多項式では、次数エムをつまみのように動かせます。"),
            ("main", "ここでは、エムイコール一から九まで、順番に動かして見ていきます。"),
            ("main", "エムイコール一なら直線。エムが二、三、四と上がるほど、曲線は少しずつ自由になります。"),
            ("main", "多項式とは、一、エックス、エックスの二乗、エックスの三乗、という材料を足し合わせて作る曲線です。"),
            ("main", "どの材料をどこまで使うかを決める数字が、次数エムです。"),
            ("main", "エムが小さいと曲線は硬い。エムが大きいと曲線はよく曲がる。"),
            ("main", "つまりエムは、曲線の自由さを決めるつまみです。"),
        ],
    },
    {
        "id": "scene07",
        "title": "最小二乗",
        "lines": [
            ("main", "曲線の良さを測るには、まず点と曲線のズレを見ます。"),
            ("main", "ズレが小さいほど、曲線は点に合っていると言えます。"),
            ("main", "ただ足すだけだと、上に外したズレと下に外したズレが打ち消し合ってしまいます。"),
            ("main", "画面では、プラスのズレとマイナスのズレをそのまま足すと、合計が小さく見えてしまいます。"),
            ("main", "そこで、ズレを二乗します。"),
            ("main", "二乗すると、上に外しても下に外してもプラスになります。"),
            ("main", "さらに、大きく外した点ほど、外し具合の点数が大きく増えます。"),
            ("main", "ズレを二乗して、全部足す。これが最小二乗の考え方です。"),
            ("main", "画面のシグマは、全部足す、という意味の省略記号です。"),
        ],
    },
    {
        "id": "scene08",
        "title": "M=1から9の比較",
        "lines": [
            ("main", "では、エムを変えると何が起きるでしょうか。"),
            ("main", "同じグラフの上で、エムイコール一から九まで、つまみを一段ずつ動かして見ます。"),
            ("main", "エムイコール一は直線です。波の形にはまだ追いつけません。"),
            ("main", "エムイコール二、三、四と上げると、曲線はだんだん点に寄っていきます。"),
            ("main", "エムイコール三あたりでは、点にもそこそこ合っていて、全体の流れもつかんでいます。"),
            ("main", "ところが、エムをさらに上げると、曲線は細かい揺れまで追いかけ始めます。"),
            ("question", "でも、九次式は全部の点を通ってるのに、どうしてダメなんですか？"),
            ("main", "いい質問です。点には合っているのに、背後の流れからは外れている。"),
            ("summary", "これが過学習です。"),
            ("main", "過学習とは、練習問題に合わせすぎて、初見問題に弱くなることです。"),
        ],
    },
    {
        "id": "scene09",
        "title": "訓練誤差とテスト誤差",
        "lines": [
            ("main", "次のグラフは、エムを変えたときの外し具合を記録したものです。"),
            ("main", "左の曲線で点とのズレを測り、その結果を右のグラフに一点ずつ置いていきます。"),
            ("main", "アールエムエス誤差は、ズレを二乗して足し、個数で割り、最後に平方根を取った値です。"),
            ("main", "画面では、ズレの本数を数えながら、その計算の流れも一緒に表示します。"),
            ("main", "練習問題だけを見ると、エムを大きくするほど外し具合は小さくなります。"),
            ("main", "これは当然です。曲線が自由になるからです。"),
            ("main", "でも、初見問題では違います。"),
            ("main", "テスト誤差は途中まで下がったあと、また大きくなってしまいます。"),
            ("summary", "練習問題で満点を取ることと、初見問題に強いことは同じではありません。"),
            ("main", "ピーアールエムエルでは、この比較のためにアールエムエス誤差を使います。"),
            ("main", "これは外し具合を、ティーと同じくらいのスケールで見やすくしたものです。"),
        ],
    },
    {
        "id": "scene10",
        "title": "係数の暴走",
        "lines": [
            ("main", "九次式が点を全部通れるのは、曲線が本当に賢くなったからではありません。"),
            ("main", "多項式には、各材料をどれだけ混ぜるかを決める係数があります。"),
            ("main", "九次式では、その係数つまみをものすごく大きくしたり、小さくしたりして、点の近くで無理やり曲げています。"),
            ("main", "これは、答えの規則を理解したというより、点の位置を丸暗記している状態に近いです。"),
            ("main", "係数が大きいこと自体が、いつも悪いわけではありません。"),
            ("main", "ただ、この例では、係数の大きさが曲線の不自然な暴れ方とつながっています。"),
        ],
    },
    {
        "id": "scene11",
        "title": "データ数を増やす",
        "lines": [
            ("main", "同じエムイコール九でも、データの数を増やすと様子が変わります。"),
            ("main", "まず点が少ない状態では、曲線は少数の点に振り回されます。"),
            ("main", "そこへ点を増やしていくと、曲線は細かな偶然よりも、全体の流れをつかみやすくなります。"),
            ("summary", "複雑なモデルを使うには、それを支えるデータも必要です。"),
        ],
    },
    {
        "id": "scene12",
        "title": "正則化",
        "lines": [
            ("main", "ここまでは、点に合うことだけを考えていました。"),
            ("main", "でも、それだけだと曲線が暴れすぎることがあります。"),
            ("main", "そこで、係数を大きくしすぎることも、外し具合の点数に足し込みます。"),
            ("main", "点からのズレを小さくしたい。でも、係数を大きくしすぎるのも避けたい。"),
            ("main", "この二つの要求のバランスを取るつまみがラムダです。"),
            ("main", "ラムダが小さすぎると、点を追いかけすぎて暴れます。"),
            ("main", "ちょうどよいと、全体の流れをつかみます。"),
            ("main", "大きすぎると、今度は曲線が硬くなりすぎます。"),
            ("main", "もう一度つまみを戻すと、曲線がまた点を追いかけ始めるのが分かります。"),
            ("summary", "ここでも大事なのは、ちょうどよい複雑さです。"),
        ],
    },
    {
        "id": "scene13",
        "title": "確率論への橋渡し",
        "lines": [
            ("main", "ここまでは、一本の曲線を選ぶ話でした。"),
            ("main", "でも実際には、データにはノイズがあります。"),
            ("main", "つまり、予測には、どれくらい不確かか、もあるはずです。"),
            ("main", "一本の線だけでなく、不確かさも同時に表したい。"),
            ("main", "そのために、次に必要になるのが確率論です。"),
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
    parser = argparse.ArgumentParser(description="Generate PRML 1.1 narration WAV files with VOICEVOX.")
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
