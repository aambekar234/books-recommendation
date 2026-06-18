"""Book catalogue used as the set of bandit arms.

Books are fetched from the Open Library subject API. The network is optional:
if it is unreachable (offline demo, rate limited) we fall back to a small
built-in catalogue so the app always works.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

OPEN_LIBRARY_SUBJECT_URL = "https://openlibrary.org/subjects/{subject}.json"
COVER_URL = "https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"


@dataclass(frozen=True)
class Book:
    """A single book — one arm of the bandit."""

    key: str
    title: str
    author: str
    subject: str
    cover_url: str | None = None

    @property
    def label(self) -> str:
        return f"{self.title} — {self.author}"


# Built-in catalogue so the demo runs without a network connection.
FALLBACK_BOOKS: list[Book] = [
    Book("ol/dune", "Dune", "Frank Herbert", "science_fiction"),
    Book("ol/neuromancer", "Neuromancer", "William Gibson", "science_fiction"),
    Book("ol/foundation", "Foundation", "Isaac Asimov", "science_fiction"),
    Book("ol/hobbit", "The Hobbit", "J.R.R. Tolkien", "fantasy"),
    Book("ol/name_of_the_wind", "The Name of the Wind", "Patrick Rothfuss", "fantasy"),
    Book("ol/mistborn", "Mistborn", "Brandon Sanderson", "fantasy"),
    Book("ol/gone_girl", "Gone Girl", "Gillian Flynn", "mystery"),
    Book("ol/da_vinci_code", "The Da Vinci Code", "Dan Brown", "mystery"),
    Book("ol/pragmatic_programmer", "The Pragmatic Programmer", "Andrew Hunt", "programming"),
    Book("ol/clean_code", "Clean Code", "Robert C. Martin", "programming"),
]


def fetch_books(subject: str, limit: int = 8, timeout: float = 6.0) -> list[Book]:
    """Fetch up to ``limit`` books for ``subject`` from Open Library.

    Falls back to the matching subset of :data:`FALLBACK_BOOKS` (or the whole
    list) if the request fails or returns nothing.
    """
    subject = subject.strip().lower().replace(" ", "_")
    try:
        resp = requests.get(
            OPEN_LIBRARY_SUBJECT_URL.format(subject=subject),
            params={"limit": limit},
            timeout=timeout,
            headers={"User-Agent": "mab-books-demo/0.1"},
        )
        resp.raise_for_status()
        works = resp.json().get("works", [])
        books = [_work_to_book(w, subject) for w in works]
        books = [b for b in books if b is not None]
        if books:
            return books[:limit]
    except (requests.RequestException, ValueError):
        pass
    return _fallback_for(subject, limit)


def fetch_mixed_catalogue(subjects: list[str], per_subject: int = 4) -> list[Book]:
    """Fetch ``per_subject`` books for each subject and combine them.

    The contextual recommender needs a catalogue spanning several genres so it
    can learn to match readers to the genre they prefer. Duplicate books (same
    key) across subjects are dropped, preserving first-seen order.
    """
    catalogue: list[Book] = []
    seen: set[str] = set()
    for subject in subjects:
        for book in fetch_books(subject, per_subject):
            if book.key not in seen:
                seen.add(book.key)
                catalogue.append(book)
    return catalogue


def _work_to_book(work: dict, subject: str) -> Book | None:
    title = work.get("title")
    if not title:
        return None
    authors = work.get("authors") or []
    author = authors[0].get("name", "Unknown") if authors else "Unknown"
    cover_id = work.get("cover_id")
    cover_url = COVER_URL.format(cover_id=cover_id) if cover_id else None
    return Book(
        key=work.get("key", title),
        title=title,
        author=author,
        subject=subject,
        cover_url=cover_url,
    )


def _fallback_for(subject: str, limit: int) -> list[Book]:
    matching = [b for b in FALLBACK_BOOKS if b.subject == subject]
    chosen = matching if matching else FALLBACK_BOOKS
    return chosen[:limit]
