import os
import os.path as osp
import sys
import subprocess
import json
import asyncio
import traceback
from datetime import datetime
from typing import Optional
from pathlib import Path
from typing import Dict, Set, List
import time
import pickle
import regex as re

import vllm     # vLLM should be imported before any 3rd-party libraries, o.w. all Pantograph REPLs are scheduled at the same CPU core.
from openai import OpenAI, AsyncOpenAI, NOT_GIVEN
from openai.types.chat import ChatCompletion
import argparse
from tqdm import tqdm
from loguru import logger
from fire import Fire

from common.constants import CORE_OPTIONS, OPEN_HEADER, BANNED_TOKENS, SYSTEM_PROMPT_SFT, CODEBLOCK_PATTERN
from common.pantograph.dataclasses import GoalState, FormalProblem, TacticInvocation, ProofSearchResult, SolutionAutoformalizationResult, TacticDraft
from common.pantograph.solving_server import PropSolvingServer
from agent.proof_search import ProofSearchResult, HammerProofSearchAgent, StepProver_NALP_LLMProofSearchAgent, StepProver_Critic_LLMProofSearchAgent, SFT_NALP_LLMProofSearchAgent, SFT_NALP_AVGGOAL_LLMProofSearchAgent
from agent.solution_autoformalization import SolutionAutoformalizer
from common.utils import remove_comments, normalize_spaces, format_solution_draft_prompt

PROVER_AGENT_DICT = {
    'hammer': HammerProofSearchAgent,
    'stepprover_vanilla': StepProver_NALP_LLMProofSearchAgent,
    'sft_vanilla': SFT_NALP_LLMProofSearchAgent,
}

def main(
    log_root: str,
    benchmark: str,
    solver_base_url: str,
    solver_api_key: str,
    solver_model_name: str,
    prover_agent: str,
    prover_base_url: Optional[str]=None,
    prover_api_key: Optional[str]=None,
    prover_model_name: Optional[str]=None,
    benchmark_root: str='data/benchmark',
    project_root: str='data/MiniF2F',
    try_num: int=8,
    temperature: float=0.7,
    max_search_trials: int=16,
    num_samples_per_trial: int=4,
    max_tokens: int=2048,
    num_concurrency: int=12,
    verbose: bool=False,
):
    saved_args = {**locals()}
    
    benchmark = benchmark.lower()
    assert benchmark in ['formal_math500', 'minif2f_solving', 'putnam_solving', 'mathodessy']
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_prefix = benchmark+'.'+'h_wg'+'.'

    os.makedirs(log_root, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level=('INFO' if not verbose else 'DEBUG'))
    logger.add(osp.join(log_root, log_prefix+now+'.log'), level='DEBUG')
    logger.info(f'Evaluating H-WG on {benchmark} with hyperparams: {saved_args}')
    log_debug = logger.info if verbose else logger.debug

    prover_client = AsyncOpenAI(
        base_url=prover_base_url,
        api_key=prover_api_key
    )
    solver_client = AsyncOpenAI(
        base_url=solver_base_url,
        api_key=solver_api_key
    )

    # Load data
    samples = []
    with open(osp.join(benchmark_root, benchmark+'.json'), 'r') as f:
        samples = json.load(f)
    if benchmark == 'formal_math500':
        samples = [
            FormalProblem(
                informal_problem=s['informal_problem'],
                informal_answer=s['informal_answer'],
                informal_solution=s['informal_solution'],
                header=s['header'],
                formal_statement=f"example\n{s['formal_problem']}\n: {s['formal_answer']}\n:= sorry",
                metainfo=dict(
                    level=s['level'],
                    subject=s['subject'],
                    annotator=s['annotator']
                )
            ) for s in samples
        ]
    elif benchmark == 'mathodessy':
        samples = [
            FormalProblem(
                informal_problem=s['informal_problem'],
                informal_answer=s['informal_answer'],
                informal_solution=s['informal_solution'],
                header=s['header'],
                formal_statement=s['formal_statement'],
                metainfo=s['metainfo']
            ) for s in samples
        ]
    elif benchmark == 'putnam_solving':
        samples = [
            FormalProblem(
                informal_problem=s['informal_statement'],
                informal_answer=s['informal_answer'],
                informal_solution=s['informal_solution'],
                header=None if len(s['opens']) == 0 else 'open ' + ' '.join(s['opens']),
                formal_statement=f"example\n{s['formal_problem']}\n: {s['formal_answer']}\n:= sorry",
                metainfo=dict(
                    problem_name=s['problem_name'],
                    tags=s['tags'],
                    annotator=s['annotator']
                )
            ) for s in samples
        ]
    elif benchmark == 'minif2f_solving':
        samples = [
            FormalProblem(
                informal_problem=s['informal_stmt'],
                informal_answer=s['informal_answer'],
                informal_solution=s['informal_proof'],
                header=None,
                formal_statement=f"example\n{s['formal_problem']}\n: {s['formal_answer']}\n:= sorry",
                metainfo=dict(
                    id=s['id'],
                    split=s['split'],
                    annotator=s['annotator'],
                    formal_statement=s['formal_statement']
                )
            ) for s in samples
        ]
    
    finished = [None for _ in range(len(samples))]
    logger.info(f"Loaded {len(samples)} samples for {benchmark} from {osp.join(benchmark_root, benchmark+'.json')}")

    proof_searcher = PROVER_AGENT_DICT[prover_agent](
        gen_client=prover_client,
        gen_model_name=prover_model_name,
        max_search_trials=max_search_trials,
        num_samples_per_trial=num_samples_per_trial,
        temperature=temperature,
        max_tokens=256,
    )
    async def search(sample: FormalProblem, tag_i: int) -> None:
        results = []
        for try_i in range(try_num):
            try:
                time_start = time.time()
                result = SolutionAutoformalizationResult(**sample.__dict__ )
                solving_server = PropSolvingServer(
                    imports=["Mathlib", "Aesop"],
                    project_path=project_root,
                    timeout=300,
                    tag=f'{tag_i}-{try_i}/{try_num}'
                )
                log_debug(f"search({tag_i}-{try_i}/{try_num}): server initialized.")
                await solving_server.load_problem_async(sample, force_parse=True)
                init_solving_state = await solving_server.init_forward_solving_state_async()
                response = (await solver_client.chat.completions.create(
                    model=solver_model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_SFT},
                        {"role": "user", "content": format_solution_draft_prompt(sample.informal_problem, str(init_solving_state.goals[0]))},
                    ],
                    max_tokens=(max_tokens if (max_tokens != NOT_GIVEN and max_tokens > 0) else NOT_GIVEN),
                    stream=False,
                    temperature=temperature
                ))
                if len(re.findall('```', response.choices[0].message.content)) == 1:
                    result.formal_solution_draft = response.choices[0].message.content[:response.choices[0].message.content.find('```')].strip()
                else:
                    parse_result = re.findall(CODEBLOCK_PATTERN, response.choices[0].message.content)
                    assert len(parse_result) == 1, f"response: {response.choices[0].message.content}"
                    result.formal_solution_draft = parse_result[0].strip()
                
                try:
                    sorry_state = await solving_server.server.goal_tactic_async(init_solving_state, 0, TacticDraft('by {\n' + result.formal_solution_draft + '\n}'))
                except Exception as e:
                    logger.debug(f'search({tag_i}-{try_i}/{try_num}): solution draft failed with {e}:\n' + traceback.format_exc())
                    results.append(result)
                    continue
                submission = await solving_server.get_submission_async(sorry_state)
                result.metainfo['submission'] = submission
                
                # Separately solve each sorry by initializing them into new search states        
                formal_gaps = sorry_state.goals[:]
                result.formal_proofs = []
                # sorry_state = sorry_unit.goal_state
                while sorry_state is not None and not sorry_state.is_solved:
                    logger.info(f'search.proof_search({tag_i}-{try_i}/{try_num}): Searching proof {len(formal_gaps)-len(sorry_state.goals)}/{len(formal_gaps)}...')
                    try:
                        search_result = await proof_searcher.search_async(
                            server=solving_server.server,
                            init_state=sorry_state,
                            tag=str(tag_i)+f'-{len(formal_gaps)-len(sorry_state.goals)}',
                            ignored_goals=sorry_state.goals[1:]
                        )
                    except Exception as e:
                        logger.error(f'search({tag_i}-{try_i}/{try_num}): Proof search failed: Exception {e}:\n{traceback.format_exc()}.')
                        if len(formal_gaps) == len(result.formal_proofs):
                            logger.critical(f'search({tag_i}-{try_i}/{try_num}): len(formal_gaps) == len(state.formal_proofs) but fails.')
                        result.formal_proofs = []
                        break
                    # If not proven, early exit
                    if not search_result.success:
                        logger.warning(f'search({tag_i}-{try_i}/{try_num}): Proof search failed.')
                        if len(formal_gaps) == len(result.formal_proofs):
                            logger.critical(f'search({tag_i}-{try_i}/{try_num}): len(formal_gaps) == len(state.formal_proofs) but fails.')
                        result.formal_proofs = []
                        break
                    result.formal_proofs.append(search_result)
                    sorry_state = search_result.final_state
                if sorry_state is None or sorry_state.is_solved:
                    logger.opt(colors=True).info(f'<green>search({tag_i}-{try_i}/{try_num}): Proof search succeeded.</green>')
                    # All sorries are successfully proven
                    result.success = True
                    # RPE Check
                    rpe_proof = await solving_server.prove_eq_async(submission)
                    result.metainfo['rpe_proof'] = rpe_proof
                    if rpe_proof is None:
                        logger.critical(f'search({tag_i}-{try_i}/{try_num}): RPE failed between "{submission}" and "{sample.formal_answer}"')
                        result.success = False
                else:
                    logger.opt(colors=True).info(f'<yellow>search({tag_i}-{try_i}/{try_num}): Proof search failed.</yellow>')

                results.append(result)
                if result.success:
                    logger.opt(colors=True).info(f'<green>search({tag_i}-{try_i}/{try_num}): succeeded at the {try_i}-th attempt in {time.time() - time_start} (s).</green>')
                    break
                else:
                    logger.info(f'search({tag_i}-{try_i}/{try_num}): failed at the {try_i}-th attempt in {time.time() - time_start} (s).')
            except:
                logger.error(f"search({tag_i}-{try_i}/{try_num}): failed because {traceback.format_exc()}")
        
        finished[tag_i] = results
        logger.info(f'search({tag_i}): finished all {try_num} attempts.')

    async def _async_main():
        pending_tasks: Set[asyncio.Task] = set()
        for i, sample in tqdm(enumerate(samples)):
            if len(pending_tasks) >= num_concurrency:
                done_tasks, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done_tasks:
                    if task.exception() is not None:
                        logger.error(f"Exception occurred: {task.exception()} {task.get_stack()}")
                        for pending_task in pending_tasks:
                            pending_task.cancel()
                        return
            pending_tasks.add(
                asyncio.create_task(
                    search(sample, i)
                )
            )
        if len(pending_tasks) > 0:
            await asyncio.wait(pending_tasks)
        await logger.complete()
    
    try:
        asyncio.run(_async_main())
    finally:
        try:
            logger.info(f"Finished search, saving at {osp.join(log_root, log_prefix+now+'.(pkl|jsonl)')}")
            with open(osp.join(log_root, log_prefix+now+'.pkl'), 'wb') as f:
                pickle.dump(finished, f)
            with open(osp.join(log_root, log_prefix+now+'.jsonl'), 'w') as f:
                for attempts in finished:
                    f.write(json.dumps(None if attempts is None else [s.serialize() for s in attempts])+'\n')
        except Exception as e:
            logger.error(traceback.format_exc())
            import pdb; pdb.set_trace()


if __name__ == '__main__':
    Fire(main)
