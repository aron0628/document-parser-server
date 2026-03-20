"""비동기 병렬 실행 유틸리티"""

import asyncio
import logging
import random
from typing import Callable, Optional

logger = logging.getLogger(__name__)


async def gather_with_semaphore(
    semaphore: asyncio.Semaphore,
    coros: list,
) -> list:
    """세마포어로 동시 실행 수를 제한하며 코루틴 리스트를 병렬 실행"""
    async def _wrap(coro):
        async with semaphore:
            return await coro
    return await asyncio.gather(*[_wrap(c) for c in coros])


async def gather_with_backpressure(
    semaphore: asyncio.Semaphore,
    coros: list,
    pause_event: asyncio.Event,
) -> list:
    """세마포어 + 글로벌 백프레셔로 동시 실행을 제한하며 코루틴 리스트를 병렬 실행

    429 Rate Limit 발생 시 pause_event.clear()로 전체 코루틴 일시 중단,
    대기 후 pause_event.set()으로 재개한다.
    """
    async def _wrap(coro):
        await pause_event.wait()  # 429 백프레셔 대기
        async with semaphore:
            return await coro
    return await asyncio.gather(*[_wrap(c) for c in coros])


def create_rate_limit_handler(pause_event: asyncio.Event):
    """429 발생 시 글로벌 백프레셔를 적용하는 콜백 생성"""
    async def on_rate_limit(wait_time: float):
        if pause_event.is_set():
            logger.warning(f"Rate limit 감지, 전체 요청 {wait_time:.1f}s 일시 중단")
            pause_event.clear()
            jittered_wait = wait_time + random.uniform(0, 1)
            await asyncio.sleep(jittered_wait)
            pause_event.set()
            logger.info("Rate limit 해제, 요청 재개")
    return on_rate_limit
