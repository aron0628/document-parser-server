"""
LangGraph Checkpointer Resume PoC

InMemorySaver를 사용하여 다양한 그래프 토폴로지에서
예외 발생 후 재개(resume) 동작을 검증한다.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List
from operator import add
import asyncio
import traceback

# ---------------------------------------------------------------------------
# 시나리오 1: Conditional loop 재개
# ---------------------------------------------------------------------------

_scenario1_fail_count = 0


class S1State(TypedDict, total=False):
    counter: int
    items: Annotated[List[str], add]


async def scenario1():
    """
    init → loop_check (conditional) → process → loop_check
    process에서 counter==2일 때 예외 발생.
    재개 시 counter 복원 + 이전 items 유지 확인.
    """
    global _scenario1_fail_count
    _scenario1_fail_count = 0

    def init(state):
        return {"counter": 0, "items": ["init"]}

    def process(state):
        global _scenario1_fail_count
        c = state["counter"] + 1
        if c == 2 and _scenario1_fail_count == 0:
            _scenario1_fail_count += 1
            raise RuntimeError("Simulated failure at counter==2")
        return {"counter": c, "items": [f"process-{c}"]}

    def loop_check(state):
        if state.get("counter", 0) >= 3:
            return "done"
        return "continue"

    g = StateGraph(S1State)
    g.add_node("init", init)
    g.add_node("process", process)
    g.add_edge(START, "init")
    g.add_edge("init", "process")
    g.add_conditional_edges("process", loop_check, {"continue": "process", "done": END})

    mem = MemorySaver()
    app = g.compile(checkpointer=mem)
    config = {"configurable": {"thread_id": "s1"}}

    print("=== 시나리오 1: Conditional loop 재개 ===")

    # 첫 실행 — 예외 발생 예상
    try:
        result = await app.ainvoke({"counter": 0, "items": []}, config)
        print(f"  첫 실행 완료 (예상 밖): {result}")
    except Exception as e:
        print(f"  첫 실행 예외: {e}")

    # 체크포인트 상태 확인
    snapshot = await app.aget_state(config)
    print(f"  체크포인트 counter: {snapshot.values.get('counter')}")
    print(f"  체크포인트 items: {snapshot.values.get('items')}")

    # 재개
    try:
        result = await app.ainvoke(None, config)
        print(f"  재개 후 결과: {result}")
        counter_ok = result.get("counter", -1) >= 3
        items_has_init = "init" in result.get("items", [])
        passed = counter_ok and items_has_init
        print(f"결과: {'PASS' if passed else 'FAIL'}")
        print(f"  세부: counter={result.get('counter')}, items={result.get('items')}")
        print(f"  counter>=3: {counter_ok}, init 유지: {items_has_init}")
    except Exception as e:
        print(f"  재개 실패: {e}")
        traceback.print_exc()
        print("결과: FAIL")
        passed = False

    print()
    return passed


# ---------------------------------------------------------------------------
# 시나리오 2: Fan-out/fan-in 재개
# ---------------------------------------------------------------------------

_scenario2_branch_b_calls = 0
_scenario2_branch_a_calls = 0
_scenario2_branch_c_calls = 0


class S2State(TypedDict, total=False):
    input: str
    results: Annotated[List[str], add]


async def scenario2():
    """
    start → fan-out [branch_a, branch_b, branch_c] → merge → END
    branch_b 첫 실행 시 예외. 재개 후 모든 결과 수집 확인.
    """
    global _scenario2_branch_b_calls, _scenario2_branch_a_calls, _scenario2_branch_c_calls
    _scenario2_branch_b_calls = 0
    _scenario2_branch_a_calls = 0
    _scenario2_branch_c_calls = 0

    def start_node(state):
        return {"input": "started", "results": ["start"]}

    def branch_a(state):
        global _scenario2_branch_a_calls
        _scenario2_branch_a_calls += 1
        return {"results": [f"a(call={_scenario2_branch_a_calls})"]}

    def branch_b(state):
        global _scenario2_branch_b_calls
        _scenario2_branch_b_calls += 1
        if _scenario2_branch_b_calls == 1:
            raise RuntimeError("branch_b first call fails")
        return {"results": [f"b(call={_scenario2_branch_b_calls})"]}

    def branch_c(state):
        global _scenario2_branch_c_calls
        _scenario2_branch_c_calls += 1
        return {"results": [f"c(call={_scenario2_branch_c_calls})"]}

    def merge(state):
        return {"results": ["merged"]}

    g = StateGraph(S2State)
    g.add_node("start_node", start_node)
    g.add_node("branch_a", branch_a)
    g.add_node("branch_b", branch_b)
    g.add_node("branch_c", branch_c)
    g.add_node("merge", merge)

    g.add_edge(START, "start_node")
    # fan-out: start_node → branch_a, branch_b, branch_c
    g.add_edge("start_node", "branch_a")
    g.add_edge("start_node", "branch_b")
    g.add_edge("start_node", "branch_c")
    # fan-in: all branches → merge
    g.add_edge("branch_a", "merge")
    g.add_edge("branch_b", "merge")
    g.add_edge("branch_c", "merge")
    g.add_edge("merge", END)

    mem = MemorySaver()
    app = g.compile(checkpointer=mem)
    config = {"configurable": {"thread_id": "s2"}}

    print("=== 시나리오 2: Fan-out/fan-in 재개 ===")

    # 첫 실행
    try:
        result = await app.ainvoke({"input": "", "results": []}, config)
        print(f"  첫 실행 완료 (예상 밖): {result}")
    except Exception as e:
        print(f"  첫 실행 예외: {e}")

    a_calls_before = _scenario2_branch_a_calls
    c_calls_before = _scenario2_branch_c_calls
    print(f"  예외 전 branch_a 호출: {a_calls_before}, branch_c 호출: {c_calls_before}")

    # 재개
    try:
        result = await app.ainvoke(None, config)
        print(f"  재개 후 결과: {result}")
        a_rerun = _scenario2_branch_a_calls > a_calls_before
        c_rerun = _scenario2_branch_c_calls > c_calls_before
        print(f"  branch_a 재실행: {a_rerun} (총 {_scenario2_branch_a_calls}회)")
        print(f"  branch_c 재실행: {c_rerun} (총 {_scenario2_branch_c_calls}회)")
        print(f"  branch_b 총 호출: {_scenario2_branch_b_calls}회")
        has_merged = "merged" in result.get("results", [])
        print(f"  merge 결과 포함: {has_merged}")
        passed = has_merged
        print(f"결과: {'PASS' if passed else 'FAIL'}")
    except Exception as e:
        print(f"  재개 실패: {e}")
        traceback.print_exc()
        print("결과: FAIL")
        passed = False

    print()
    return passed


# ---------------------------------------------------------------------------
# 시나리오 3: Conditional edge 재개
# ---------------------------------------------------------------------------

_scenario3_step3a_calls = 0
_scenario3_step2_calls = 0


class S3State(TypedDict, total=False):
    value: str
    route: str
    log: Annotated[List[str], add]


async def scenario3():
    """
    step_1 → step_2 → conditional_edge(check) → step_3a or step_3b → END
    step_3a 첫 실행 시 예외. 재개 시 step_2 재실행 없이 step_3a부터 시작 확인.
    """
    global _scenario3_step3a_calls, _scenario3_step2_calls
    _scenario3_step3a_calls = 0
    _scenario3_step2_calls = 0

    def step_1(state):
        return {"value": "one", "route": "a", "log": ["step1"]}

    def step_2(state):
        global _scenario3_step2_calls
        _scenario3_step2_calls += 1
        return {"log": [f"step2(call={_scenario3_step2_calls})"]}

    def check(state):
        return state.get("route", "a")

    def step_3a(state):
        global _scenario3_step3a_calls
        _scenario3_step3a_calls += 1
        if _scenario3_step3a_calls == 1:
            raise RuntimeError("step_3a first call fails")
        return {"log": [f"step3a(call={_scenario3_step3a_calls})"]}

    def step_3b(state):
        return {"log": ["step3b"]}

    g = StateGraph(S3State)
    g.add_node("step_1", step_1)
    g.add_node("step_2", step_2)
    g.add_node("step_3a", step_3a)
    g.add_node("step_3b", step_3b)

    g.add_edge(START, "step_1")
    g.add_edge("step_1", "step_2")
    g.add_conditional_edges("step_2", check, {"a": "step_3a", "b": "step_3b"})
    g.add_edge("step_3a", END)
    g.add_edge("step_3b", END)

    mem = MemorySaver()
    app = g.compile(checkpointer=mem)
    config = {"configurable": {"thread_id": "s3"}}

    print("=== 시나리오 3: Conditional edge 재개 ===")

    # 첫 실행
    try:
        result = await app.ainvoke({"value": "", "route": "a", "log": []}, config)
        print(f"  첫 실행 완료 (예상 밖): {result}")
    except Exception as e:
        print(f"  첫 실행 예외: {e}")

    step2_before = _scenario3_step2_calls
    print(f"  예외 전 step_2 호출: {step2_before}")

    # 재개
    try:
        result = await app.ainvoke(None, config)
        print(f"  재개 후 결과: {result}")
        step2_rerun = _scenario3_step2_calls > step2_before
        print(f"  step_2 재실행: {step2_rerun} (총 {_scenario3_step2_calls}회)")
        print(f"  step_3a 총 호출: {_scenario3_step3a_calls}회")
        has_step3a = any("step3a" in x for x in result.get("log", []))
        no_step2_rerun = not step2_rerun
        passed = has_step3a and no_step2_rerun
        print(f"결과: {'PASS' if passed else 'FAIL'}")
        print(f"  세부: step_3a 결과 포함={has_step3a}, step_2 미재실행={no_step2_rerun}")
    except Exception as e:
        print(f"  재개 실패: {e}")
        traceback.print_exc()
        print("결과: FAIL")
        passed = False

    print()
    return passed


# ---------------------------------------------------------------------------
# 시나리오 4: 부분 브랜치 완료 상태 재개
# ---------------------------------------------------------------------------

_scenario4_fast_calls = 0
_scenario4_slow_calls = 0


class S4State(TypedDict, total=False):
    data: str
    log: Annotated[List[str], add]


async def scenario4():
    """
    start → fan-out [fast_branch, slow_branch]
    fast_branch → merge → END
    slow_branch → merge → END
    slow_branch 첫 실행 시 예외. fast_branch는 이미 완료.
    재개 시 fast_branch 재실행 여부 확인.
    """
    global _scenario4_fast_calls, _scenario4_slow_calls
    _scenario4_fast_calls = 0
    _scenario4_slow_calls = 0

    def start_node(state):
        return {"data": "started", "log": ["start"]}

    def fast_branch(state):
        global _scenario4_fast_calls
        _scenario4_fast_calls += 1
        return {"log": [f"fast(call={_scenario4_fast_calls})"]}

    def slow_branch(state):
        global _scenario4_slow_calls
        _scenario4_slow_calls += 1
        if _scenario4_slow_calls == 1:
            raise RuntimeError("slow_branch first call fails")
        return {"log": [f"slow(call={_scenario4_slow_calls})"]}

    def merge(state):
        return {"log": ["merged"]}

    g = StateGraph(S4State)
    g.add_node("start_node", start_node)
    g.add_node("fast_branch", fast_branch)
    g.add_node("slow_branch", slow_branch)
    g.add_node("merge", merge)

    g.add_edge(START, "start_node")
    g.add_edge("start_node", "fast_branch")
    g.add_edge("start_node", "slow_branch")
    g.add_edge("fast_branch", "merge")
    g.add_edge("slow_branch", "merge")
    g.add_edge("merge", END)

    mem = MemorySaver()
    app = g.compile(checkpointer=mem)
    config = {"configurable": {"thread_id": "s4"}}

    print("=== 시나리오 4: 부분 브랜치 완료 상태 재개 ===")

    # 첫 실행
    try:
        result = await app.ainvoke({"data": "", "log": []}, config)
        print(f"  첫 실행 완료 (예상 밖): {result}")
    except Exception as e:
        print(f"  첫 실행 예외: {e}")

    fast_before = _scenario4_fast_calls
    print(f"  예외 전 fast_branch 호출: {fast_before}")

    # 재개
    try:
        result = await app.ainvoke(None, config)
        print(f"  재개 후 결과: {result}")
        fast_rerun = _scenario4_fast_calls > fast_before
        print(f"  fast_branch 재실행: {fast_rerun} (총 {_scenario4_fast_calls}회)")
        print(f"  slow_branch 총 호출: {_scenario4_slow_calls}회")
        has_merged = "merged" in result.get("log", [])
        print(f"  merge 결과 포함: {has_merged}")
        passed = has_merged
        print(f"결과: {'PASS' if passed else 'FAIL'}")
        print(f"  세부: fast_branch 재실행={fast_rerun}")
    except Exception as e:
        print(f"  재개 실패: {e}")
        traceback.print_exc()
        print("결과: FAIL")
        passed = False

    print()
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    results = {}

    results["1. Conditional loop 재개"] = await scenario1()
    results["2. Fan-out/fan-in 재개"] = await scenario2()
    results["3. Conditional edge 재개"] = await scenario3()
    results["4. 부분 브랜치 완료 재개"] = await scenario4()

    print("=" * 60)
    print("전체 요약")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    total = len(results)
    passed_count = sum(1 for v in results.values() if v)
    print(f"\n  총 {total}개 중 {passed_count}개 PASS, {total - passed_count}개 FAIL")


if __name__ == "__main__":
    asyncio.run(main())
