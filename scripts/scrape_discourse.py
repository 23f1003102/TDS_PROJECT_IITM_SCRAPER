# scripts/scrape_discourse.py

import os
import json
from datetime import datetime
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_ID = 34
CATEGORY_JSON_URL = f"{BASE_URL}/c/courses/tds-kb/{CATEGORY_ID}.json"
AUTH_STATE_FILE = "auth.json"
DATE_FROM = datetime(2025, 1, 1)
DATE_TO = datetime(2025, 4, 14)

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")

def login_and_save_auth(playwright):
    print("🔐 Login required. Opening browser...")
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto(f"{BASE_URL}/login")
    print("➡️ Log in using Google (IITM email), then press ▶️ in Playwright.")
    page.pause()
    context.storage_state(path=AUTH_STATE_FILE)
    print("✅ Login state saved.")
    browser.close()

def is_authenticated(page):
    try:
        page.goto(CATEGORY_JSON_URL, timeout=10000)
        page.wait_for_selector("pre", timeout=5000)
        json.loads(page.inner_text("pre"))
        return True
    except (TimeoutError, json.JSONDecodeError):
        return False

def scrape_posts(playwright):
    os.makedirs("data", exist_ok=True)
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(storage_state=AUTH_STATE_FILE)
    page = context.new_page()

    all_topics = []
    page_num = 0

    while True:
        paginated_url = f"{CATEGORY_JSON_URL}?page={page_num}"
        print(f"📦 Fetching page {page_num}")
        page.goto(paginated_url)

        try:
            data = json.loads(page.inner_text("pre"))
        except:
            data = json.loads(page.content())

        topics = data.get("topic_list", {}).get("topics", [])
        if not topics:
            break

        all_topics.extend(topics)
        page_num += 1

    print(f"✅ Found {len(all_topics)} topics")

    filtered_posts = []

    for topic in all_topics:
        created_at = parse_date(topic["created_at"])
        if DATE_FROM <= created_at <= DATE_TO:
            topic_url = f"{BASE_URL}/t/{topic['slug']}/{topic['id']}.json"
            page.goto(topic_url)

            try:
                topic_data = json.loads(page.inner_text("pre"))
            except:
                topic_data = json.loads(page.content())

            posts = topic_data.get("post_stream", {}).get("posts", [])
            for post in posts:
                filtered_posts.append({
                    "topic_id": topic["id"],
                    "topic_title": topic.get("title"),
                    "author": post["username"],
                    "created_at": post["created_at"],
                    "content": BeautifulSoup(post["cooked"], "html.parser").get_text()
                })

    with open("data/discourse_posts.json", "w", encoding="utf-8") as f:
        json.dump(filtered_posts, f, indent=2)

    print(f"✅ Saved {len(filtered_posts)} posts")
    browser.close()

def main():
    with sync_playwright() as p:
        if not os.path.exists(AUTH_STATE_FILE):
            login_and_save_auth(p)
        else:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(storage_state=AUTH_STATE_FILE)
            page = context.new_page()
            if not is_authenticated(page):
                browser.close()
                login_and_save_auth(p)
            else:
                browser.close()
        scrape_posts(p)

if __name__ == "__main__":
    main()
