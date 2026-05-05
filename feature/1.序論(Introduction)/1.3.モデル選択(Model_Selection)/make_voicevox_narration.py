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

PAUSE_SECONDS = 0.30

SCENES = [
    {
        "id": "scene01",
        "title": "モデル選択とは",
        "lines": [
            ("main", "前回の多項式曲線フィッティングでは、次数エムを変えると曲線の自由さが変わりました。"),
            ("main", "エムが小さすぎると、データの流れを追えません。"),
            ("main", "エムが大きすぎると、訓練データに合わせすぎます。"),
            ("question", "では、どのエムを選べばよいのでしょうか。"),
            ("summary", "この、モデルの複雑さをどう選ぶかが、モデル選択です。"),
            ("main", "目的は、訓練データで一番よく見えるモデルではなく、まだ見ていないデータに強いモデルを選ぶことです。"),
        ],
    },
    {
        "id": "scene02",
        "title": "訓練誤差だけでは選べない",
        "lines": [
            ("main", "訓練データだけを見ると、エムを大きくするほど誤差は下がります。"),
            ("main", "曲線が自由になれば、見えている点には合わせやすいからです。"),
            ("main", "でも、初見データの誤差は同じ動きにはなりません。"),
            ("main", "最初は下がりますが、複雑にしすぎると、また上がります。"),
            ("main", "訓練誤差だけで選ぶと、つい一番複雑なモデルを選んでしまいます。"),
            ("summary", "それは、練習問題に一番合わせたモデルであって、初見問題に一番強いモデルとは限りません。"),
        ],
    },
    {
        "id": "scene03",
        "title": "Validation set と Test set",
        "lines": [
            ("main", "データが十分にあるなら、使い道を分ける方法があります。"),
            ("main", "まずトレイン、訓練データで、候補のモデルをそれぞれ学習します。"),
            ("main", "次にバリデーション、検証データで、どの複雑さがよいかを比べます。"),
            ("main", "そして最後にテストデータで、選び終わったモデルの実力を一度だけ測ります。"),
            ("summary", "大事なのは、テストデータを選択の途中で何度も見ないことです。"),
            ("main", "何度も見ながら調整すると、検証データやテストデータにも合わせすぎてしまうからです。"),
        ],
    },
    {
        "id": "scene04",
        "title": "Cross-validation",
        "lines": [
            ("main", "でも、いつもデータがたくさんあるとは限りません。"),
            ("main", "検証用に多く取り分けると、学習に使えるデータが減ります。"),
            ("main", "検証用が少なすぎると、評価がたまたまに左右されます。"),
            ("summary", "この困りごとへの代表的な答えが、交差検証です。"),
            ("main", "たとえば四分割なら、四つのうち三つで学習し、残り一つで評価します。"),
            ("main", "評価に使うブロックを入れ替えて、これを四回繰り返します。"),
            ("main", "最後に四つの点数を平均して、複雑さを比べます。"),
        ],
    },
    {
        "id": "scene05",
        "title": "Leave-one-out",
        "lines": [
            ("main", "さらにデータが少ないときは、エスをデータ数エヌと同じにすることもあります。"),
            ("main", "これは、ひとつだけ取り分けて、残り全部で学習する方法です。"),
            ("summary", "名前は、リーブワンアウト。"),
            ("main", "学習にはほとんど全部のデータを使えます。"),
            ("main", "その代わり、データがエヌ個なら、エヌ回の学習が必要になります。"),
        ],
    },
    {
        "id": "scene06",
        "title": "探索コスト",
        "lines": [
            ("main", "交差検証には弱点もあります。"),
            ("main", "まず、エス分割なら、学習回数はだいたいエス倍になります。"),
            ("main", "さらに、調整したい複雑さのつまみが一つとは限りません。"),
            ("main", "次数エム、正則化の強さラムダ、ほかの設定。"),
            ("main", "つまみが増えると、試す組み合わせはすぐに増えます。"),
            ("summary", "重いモデルでは、この探索だけで大きな計算コストになります。"),
        ],
    },
    {
        "id": "scene07",
        "title": "情報量基準",
        "lines": [
            ("main", "そこで、検証データを使わず、訓練データだけから選べないか、という方向も考えられます。"),
            ("main", "歴史的には、情報量基準と呼ばれる方法が提案されました。"),
            ("main", "エーアイシーはその一つです。"),
            ("main", "よく当てはまるほど、対数尤度は大きくなります。"),
            ("main", "一方で、調整できるパラメータが多いモデルには、複雑さの分だけペナルティを入れます。"),
            ("summary", "つまり、当てはまりの良さから、複雑すぎる分を差し引く考え方です。"),
        ],
    },
    {
        "id": "scene08",
        "title": "1.4 への橋渡し",
        "lines": [
            ("main", "モデル選択の中心にある問いは一つです。"),
            ("main", "見えているデータへの当てはまりと、まだ見ていないデータへの強さを、どう区別するか。"),
            ("main", "検証データを分ける方法。交差検証でデータを使い回す方法。複雑さのペナルティを入れる方法。"),
            ("main", "そして、ピーアールエムエルがこの先で重視する、ベイズ的な方法。"),
            ("summary", "どれも、過学習の楽観をどう抑えるか、という問題への答えです。"),
            ("main", "次は、入力の数が増えたときに何が起きるのか、次元の呪いへ進みます。"),
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
    query["postPhonemeLength"] = 0.20
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


def synthesize_scene(base_url: str, scene: dict[str, object], overwrite: bool) -> dict[str, object]:
    scene_id = str(scene["id"])
    output_path = OUTPUT_DIR / f"{scene_id}.wav"
    if output_path.exists() and not overwrite:
        with wave.open(str(output_path), "rb") as existing:
            duration = existing.getnframes() / existing.getframerate()
        return {"id": scene_id, "title": scene["title"], "path": str(output_path), "duration": duration, "skipped": True}

    expected_params: tuple[int, int, int] | None = None
    with wave.open(str(output_path), "wb") as output:
        for speaker_key, text in scene["lines"]:  # type: ignore[index]
            wav_bytes = synthesize(base_url, str(speaker_key), str(text))
            expected_params = append_wav_bytes(output, wav_bytes, expected_params)
            append_silence(output, expected_params, PAUSE_SECONDS)
            time.sleep(0.08)

    with wave.open(str(output_path), "rb") as generated:
        duration = generated.getnframes() / generated.getframerate()
    return {"id": scene_id, "title": scene["title"], "path": str(output_path), "duration": duration, "skipped": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VOICEVOX narration WAV files for PRML 1.3.")
    parser.add_argument("--base-url", default="http://127.0.0.1:50021", help="VOICEVOX Engine base URL")
    parser.add_argument("--from-scene", default=None, help="Resume from a scene id, e.g. scene04")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing WAV files")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = SCENES
    if args.from_scene:
        scene_ids = [scene["id"] for scene in SCENES]
        if args.from_scene not in scene_ids:
            raise SystemExit(f"Unknown scene id: {args.from_scene}")
        selected = SCENES[scene_ids.index(args.from_scene) :]

    manifest: list[dict[str, object]] = []
    manifest_path = OUTPUT_DIR / "manifest.json"
    if manifest_path.exists() and args.from_scene:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected_ids = {scene["id"] for scene in selected}
        manifest = [item for item in manifest if item.get("id") not in selected_ids]

    for scene in selected:
        result = synthesize_scene(args.base_url, scene, overwrite=args.overwrite)
        status = "skip" if result["skipped"] else "generated"
        print(f"{status}: {result['id']} {result['duration']:.2f}s")
        manifest.append(result)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

