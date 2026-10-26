"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask --chat
    uv run python -m app.ask --chat --session lecture01

通常は1回の質問を処理する。--chatを付けると同じプロセス内で会話を続けられ、
--sessionを付けると会話履歴を.sessions/に保存して次回起動時にも再開できる。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import json
from pathlib import Path

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
SESSION_DIR = Path(__file__).resolve().parent.parent / ".sessions"


def session_path(session_name: str) -> Path:
    """セッション名から保存先を作る。ディレクトリ移動は許可しない。"""
    if not session_name or Path(session_name).name != session_name or session_name in {".", ".."}:
        raise ValueError("セッション名にはディレクトリを含められません")
    return SESSION_DIR / f"{session_name}.json"


def load_history(session_name: str) -> list[dict[str, object]]:
    """保存済みのチャット履歴を読み込む。未作成なら空の履歴を返す。"""
    path = session_path(session_name)
    if not path.exists():
        return []
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list):
        raise ValueError(f"履歴ファイルの形式が不正です: {path}")
    return history


def save_history(session_name: str, history: list[dict[str, object]]) -> None:
    """チャット履歴をJSONに保存する。"""
    path = session_path(session_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_usage(response) -> None:
    usage = response.usage_metadata
    if usage is None:
        print("[usage] トークン数の情報はありません")
    else:
        print(f"[usage] 入力トークン数: {usage.prompt_token_count}")
        print(f"[usage] 出力トークン数: {usage.candidates_token_count}")
        print(f"[usage] 合計トークン数: {usage.total_token_count}")


def ask(question: str, chat=None) -> str:
    """質問を送る。chatを渡すと、そのチャットの履歴を引き継ぐ。"""
    if chat is None:
        client = genai.Client(api_key=get_api_key())
        chat = client.chats.create(model=MODEL)
    response = chat.send_message(question)
    print_usage(response)
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="質問を繰り返し受け付ける")
    parser.add_argument("--session", help="会話履歴を保存・復元するセッション名")
    args = parser.parse_args()

    if args.chat and args.question:
        parser.error("--chat と質問は同時に指定できません")
    if not args.chat and args.question is None:
        args.question = DEFAULT_QUESTION

    print(f"[model] {MODEL}")
    try:
        client = genai.Client(api_key=get_api_key())
        history = load_history(args.session) if args.session else []
        chat = client.chats.create(model=MODEL, history=history)

        if args.chat:
            if args.session:
                print(f"[session] {args.session}")
            print("質問を入力してください。終了するには exit または quit を入力します。")
            while True:
                try:
                    question = input("\n質問> ").strip()
                except EOFError:
                    print()
                    break
                if question.lower() in {"exit", "quit"}:
                    break
                if not question:
                    continue
                print(f"[質問] {question}\n")
                answer = ask(question, chat)
                print(answer)
                if args.session:
                    history.extend(
                        [
                            {"role": "user", "parts": [{"text": question}]},
                            {"role": "model", "parts": [{"text": answer}]},
                        ]
                    )
                    save_history(args.session, history)
        else:
            print(f"[質問] {args.question}\n")
            answer = ask(args.question, chat)
            print(answer)
            if args.session:
                history.extend(
                    [
                        {"role": "user", "parts": [{"text": args.question}]},
                        {"role": "model", "parts": [{"text": answer}]},
                    ]
                )
                save_history(args.session, history)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
