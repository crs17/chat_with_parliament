#!/usr/bin/env python3
"""Load Enhedslisten SPA (hvad-mener-enhedslisten-om) and print all links to politisk-abc.
Run locally to get the full list for PARTY_MANIFESTS['Ø']: uv run python scripts/fetch_enhedslisten_politisk_abc_links.py
Currently PARTY_MANIFESTS uses 7 verified slugs; run this script to discover more."""

from playwright.sync_api import sync_playwright

URL = "https://enhedslisten.dk/det-vil-vi/hvad-mener-enhedslisten-om/#/"

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(URL, wait_until="load", timeout=30_000)
            # Let SPA render topic list (links may appear after JS)
            page.wait_for_timeout(3000)
            # All links that point to politisk-abc
            links = page.eval_on_selector_all(
                'a[href*="politisk-abc"]',
                """nodes => nodes.map(a => a.getAttribute('href'))"""
            )
            # Normalize to full URLs and dedupe
            base = "https://enhedslisten.dk"
            seen = set()
            for href in links:
                if not href:
                    continue
                href = href.strip()
                if href.startswith("/"):
                    href = base + href
                elif not href.startswith("http"):
                    continue
                if "politisk-abc" in href and href not in seen:
                    seen.add(href)
                    print(href)
        finally:
            browser.close()

if __name__ == "__main__":
    main()
