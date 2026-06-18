import requests

from mab_books import books as books_mod
from mab_books.books import fetch_books, FALLBACK_BOOKS


def test_fetch_books_falls_back_on_network_error(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(books_mod.requests, "get", boom)
    result = fetch_books("science_fiction", limit=3)
    assert 1 <= len(result) <= 3
    assert all(b.subject == "science_fiction" for b in result)


def test_fetch_books_uses_full_catalogue_for_unknown_subject(monkeypatch):
    monkeypatch.setattr(
        books_mod.requests, "get",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()),
    )
    result = fetch_books("unknown_subject_xyz", limit=5)
    assert len(result) == 5  # falls back to first 5 of full catalogue


def test_fetch_books_parses_open_library_payload(monkeypatch):
    payload = {
        "works": [
            {
                "key": "/works/OL1W",
                "title": "Test Book",
                "authors": [{"name": "Jane Doe"}],
                "cover_id": 12345,
            }
        ]
    }

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(books_mod.requests, "get", lambda *a, **k: FakeResp())
    result = fetch_books("fantasy", limit=5)
    assert result[0].title == "Test Book"
    assert result[0].author == "Jane Doe"
    assert "12345" in result[0].cover_url


def test_book_label():
    b = FALLBACK_BOOKS[0]
    assert b.title in b.label and b.author in b.label
