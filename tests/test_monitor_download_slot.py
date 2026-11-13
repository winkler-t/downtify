"""Playlist Monitor downloads share the "Parallel downloads" limit.

Watch checks (scheduled sweeps, and the first check of a newly added
watch, which runs as its own background task) used to download outside
the queue's semaphore, so "Parallel downloads = 1" still ran one download
per active watch on top of the queue. These tests count how many
downloads run at the same time; they would see 2 if a path went back to
skipping the shared slot.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from downtify import api, download_queue
from downtify import monitor as monitor_mod

_SONG = {
    'song_id': 'sp1',
    'name': 'Song',
    'artists': ['Artist'],
    'youtube_id': 'abc123def45',
    'album_name': 'Album',
    'cover_url': '',
}


class _CountingDownloader:
    """Records the peak number of downloads running at the same time."""

    def __init__(self, hold: float = 0.15) -> None:
        self._lock = threading.Lock()
        self._running = 0
        self.peak = 0
        self.calls = 0
        self._hold = hold

    def download(self, song, progress, subdir=None):  # noqa: ARG002
        with self._lock:
            self._running += 1
            self.calls += 1
            self.peak = max(self.peak, self._running)
        try:
            time.sleep(self._hold)
        finally:
            with self._lock:
                self._running -= 1
        return f'{song["song_id"]}.mp3'


class _FakeMonitorDB:
    def __init__(self) -> None:
        self.marked: list[str] = []

    def mark_track_downloaded(self, playlist_id, song_id, filename, **_kw):
        self.marked.append(filename)


async def _noop_broadcast(_msg: dict[str, Any]) -> None:
    return None


def _watch(watch_id: int) -> Any:
    return SimpleNamespace(id=watch_id, name=f'Watch {watch_id}')


async def _fetch(downloader, db, watch_id: int, song_id: str) -> str:
    return await monitor_mod._fetch_watch_track(
        {**_SONG, 'song_id': song_id},
        _watch(watch_id),
        db,
        downloader,
        _noop_broadcast,
        asyncio.get_running_loop(),
        subdir='',
        initial=True,
    )


@pytest.fixture
def one_slot(monkeypatch):
    """A "Parallel downloads = 1" limit wired the way the app wires it."""

    holder: dict[str, Any] = {}

    def _install() -> asyncio.Semaphore:
        sem = asyncio.Semaphore(1)
        holder['sem'] = sem
        monkeypatch.setattr(api.state, 'download_semaphore', sem)
        return sem

    monkeypatch.setattr(
        download_queue,
        '_slot_source',
        lambda: api.state.download_semaphore,
    )
    return _install


def test_api_registers_the_queue_semaphore_as_slot_source(monkeypatch):
    sem = object()
    monkeypatch.setattr(api.state, 'download_semaphore', sem)
    assert download_queue._slot_source is not None
    assert download_queue._slot_source() is sem


def test_download_slot_without_a_limit_is_a_noop(monkeypatch):
    monkeypatch.setattr(download_queue, '_slot_source', None)

    async def _run() -> bool:
        async with download_queue.download_slot():
            return True

    assert asyncio.run(_run())


def test_download_slot_bounds_concurrent_holders(one_slot):
    async def _run() -> int:
        one_slot()
        running = 0
        peak = 0

        async def _hold() -> None:
            nonlocal running, peak
            async with download_queue.download_slot():
                running += 1
                peak = max(peak, running)
                await asyncio.sleep(0.02)
                running -= 1

        await asyncio.gather(*(_hold() for _ in range(4)))
        return peak

    assert asyncio.run(_run()) == 1


def test_parallel_watch_checks_respect_one_parallel_download(one_slot):
    # Two watches checked at the same time (e.g. two artists just added,
    # each starting its own initial check) must not download in parallel.
    downloader = _CountingDownloader()
    db = _FakeMonitorDB()

    async def _run() -> list[str]:
        one_slot()
        return await asyncio.gather(
            _fetch(downloader, db, 1, 'a'),
            _fetch(downloader, db, 2, 'b'),
            _fetch(downloader, db, 3, 'c'),
        )

    results = asyncio.run(_run())

    assert sorted(results) == ['a.mp3', 'b.mp3', 'c.mp3']
    assert downloader.calls == 3
    assert downloader.peak == 1
    assert sorted(db.marked) == ['a.mp3', 'b.mp3', 'c.mp3']


def test_watch_check_and_queue_share_the_same_slot(one_slot, monkeypatch):
    downloader = _CountingDownloader()
    db = _FakeMonitorDB()
    monkeypatch.setattr(api.state, 'downloader', downloader)
    monkeypatch.setattr(api.state, 'loop', None)
    monkeypatch.setattr(api.state.connections, 'broadcast', _noop_broadcast)

    async def _run() -> list[Any]:
        one_slot()
        return await asyncio.gather(
            api._run_download({**_SONG, 'song_id': 'queued'}, 'job-q'),
            _fetch(downloader, db, 1, 'watched'),
        )

    queued, watched = asyncio.run(_run())

    assert queued == 'queued.mp3'
    assert watched == 'watched.mp3'
    assert downloader.peak == 1


def test_higher_limit_still_allows_parallel_watch_downloads(monkeypatch):
    # The fix must not serialise users who chose more than one slot.
    downloader = _CountingDownloader(hold=0.3)
    db = _FakeMonitorDB()

    async def _run() -> None:
        sem = asyncio.Semaphore(3)
        monkeypatch.setattr(api.state, 'download_semaphore', sem)
        monkeypatch.setattr(
            download_queue,
            '_slot_source',
            lambda: api.state.download_semaphore,
        )
        await asyncio.gather(
            _fetch(downloader, db, 1, 'a'),
            _fetch(downloader, db, 2, 'b'),
            _fetch(downloader, db, 3, 'c'),
        )

    asyncio.run(_run())

    assert downloader.peak == 3
