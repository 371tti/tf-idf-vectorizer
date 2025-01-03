import requests
import json
import os
from time import sleep
from datetime import datetime, timedelta

# Wikimedia Pageviews APIのURL
PAGEVIEWS_API_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/ja.wikipedia/all-access/"

# JSON出力ディレクトリ
OUTPUT_DIR = "popular_wikipedia_articles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ファイル名として使用可能な文字のみを残す
def sanitize_filename(filename):
    import re
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

# 現在の年月を取得し、直近の月を使用
def get_latest_available_date():
    today = datetime.today()
    # データが遅れる場合を考慮し、1ヶ月前の日付を使用
    last_month = today - timedelta(days=30)
    return last_month.year, last_month.month

# 人気な記事リストを取得する関数
def fetch_popular_articles(year, month):
    url = f"{PAGEVIEWS_API_URL}{year}/{str(month).zfill(2)}/all-days"
    headers = {
        "User-Agent": "YourAppName/1.0 (your.email@example.com)"  # 適切に変更
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    articles = data.get("items", [])[0].get("articles", [])
    return articles[:100]  # 上位100件を取得

# 記事内容を取得する関数
def fetch_article_content(title):
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "titles": title,
        "format": "json",
    }
    headers = {
        "User-Agent": "YourAppName/1.0 (your.email@example.com)"
    }
    response = requests.get("https://ja.wikipedia.org/w/api.php", params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))  # 最初の要素を取得
    if not page or "missing" in page:
        return None, None
    title = page.get("title", "Unknown")
    content = page.get("revisions", [{}])[0].get("*", "")
    return title, content

# 記事をJSON形式で保存する関数
def save_article_to_json(title, content, rank):
    safe_title = sanitize_filename(title)
    filename = os.path.join(OUTPUT_DIR, f"{rank}_{safe_title}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"rank": rank, "title": title, "content": content}, f, ensure_ascii=False, indent=4)

# メイン処理
def main():
    year, month = get_latest_available_date()

    # 人気な記事リストを取得
    print(f"Fetching popular articles for {year}/{month}")
    articles = fetch_popular_articles(year, month)

    for rank, article in enumerate(articles, start=1):
        title = article["article"]
        print(f"Fetching article: {title} (Rank: {rank})")

        try:
            # 記事内容を取得
            title, content = fetch_article_content(title)
            if title and content:
                save_article_to_json(title, content, rank)
                print(f"Saved: {title} (Rank: {rank})")
            else:
                print(f"Article {title} does not exist.")
        except Exception as e:
            print(f"Error fetching article {title}: {e}")
            sleep(1)

        # API制限を考慮して少し待機
        sleep(0.1)

    print("Finished downloading popular articles.")

if __name__ == "__main__":
    main()
