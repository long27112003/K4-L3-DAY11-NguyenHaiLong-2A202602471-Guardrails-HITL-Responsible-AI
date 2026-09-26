"""Bounded CP4 API pacing and transient-error recovery."""
import asyncio
import logging
import time


class AttackCaller:
    def __init__(self, interval=13.0, timeout=45.0, retries=1):
        self.interval, self.timeout, self.retries = interval, timeout, retries
        self.next_call = 0.0

    async def call(self, operation):
        for attempt in range(self.retries + 1):
            wait = max(0, self.next_call - time.monotonic())
            if wait:
                print(f'Waiting {wait:.0f}s for API quota...', flush=True)
                await asyncio.sleep(wait)
            self.next_call = time.monotonic() + self.interval
            try:
                # ADK prints the same traceback at several levels; retain errors in results.
                logger = logging.getLogger('google.adk')
                previous = logger.level
                logger.setLevel(logging.CRITICAL)
                try:
                    return await asyncio.wait_for(operation(), timeout=self.timeout)
                finally:
                    logger.setLevel(previous)
            except Exception as exc:
                status = error_status(exc)
                if status not in (429, 503) or attempt == self.retries:
                    raise
                delay = 30 if status == 429 else 5
                self.next_call = max(self.next_call, time.monotonic() + delay)
                print(f'API {status}; retry {attempt + 1}/{self.retries} in {delay}s.', flush=True)


def error_status(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        code = getattr(exc, 'status_code', None) or getattr(exc, 'code', None)
        for status in (429, 503):
            if code == status or str(status) in str(exc):
                return status
        exc = exc.__cause__ or exc.__context__
    return None


_callers = {}


def get_attack_caller(provider, model):
    key = (provider, model)
    if key not in _callers:
        _callers[key] = AttackCaller(interval=13 if provider == 'gemini' else 0)
    return _callers[key]
