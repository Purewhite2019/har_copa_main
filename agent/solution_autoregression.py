from abc import abstractmethod
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Dict, Set, Any
import collections, unittest
import heapq
import asyncio
import traceback
import json
import regex as re

from loguru import logger
import networkx as nx
from openai import AsyncOpenAI, NOT_GIVEN
from openai.types.chat.chat_completion import Choice
from easydict import EasyDict
import vllm
from transformers import AutoTokenizer

from common.constants import BANNED_TOKENS, CODEBLOCK_PATTERN
from common.pantograph.server import Server, TacticFailure, ServerError
from common.pantograph.solving_server import PropSolvingServer
from common.pantograph.dataclasses import TacticHave, TacticLet, TacticDraft, Tactic, GoalState, Goal, FormalProblem
from common.utils import zip_strict, remove_comments, format_forward_solution_step_prompt, normalize_spaces, extract_code
from agent.proof_search import ProofSearchResult, ProofSearchAgent


@dataclass(frozen=True)
class SolutionStep:
    step: str
    proofs: List[Tuple[int, str]]

    def serialize(self) -> Dict:
        return {
            'step': self.step,
            'proofs': self.proofs
        } 

@dataclass(frozen=True)
class SolutionAutoregressionResult: # (state_i) -> SolutionSearcher -> (step_i) -> ProofSearcher -> (proof_i)
    duration: float = float('inf')
    success: bool = False
    solution: List[SolutionStep] = field(default_factory=list)  # (step_i, proof_i)
    states: List[GoalState] = field(default_factory=list)   # (state_i)
    submission: Optional[str] = None
    backward_proof: Optional[List[Tuple[int, str]]] = None
    rpe_proof: Optional[str] = None
    cost: int=float('inf')


    def serialize(self) -> Dict:
        return {
            'duration': self.duration,
            'success': self.success,
            'solution': [s.serialize() for s in self.solution],
            'states': [s.serialize() for s in self.states],
            'submission': self.submission,
            'backward_proof': self.backward_proof,
            'rpe_proof': self.rpe_proof,
            'cost': self.cost
        }

def to_simple_str(state: Union[Goal, GoalState]) -> str:
    if isinstance(state, Goal):
        return json.dumps([state.serialize()])
    elif isinstance(state, GoalState):
        return json.dumps([g.serialize() for g in state.goals])
    else:
        raise NotImplementedError(f'to_simple_str() is not implemented for {type(state)}.')

class SolutionAutoregressionAgent:
    """
    A template autoregessive solution search agent for solution search.
    """

    def __init__(self, max_search_trials: int, proof_searcher: Optional[ProofSearchAgent]) -> None:
        self.max_search_trials = max_search_trials
        self.proof_searcher = proof_searcher

    @abstractmethod
    async def gen_step_async(
            self,
            sample: FormalProblem,
            state: GoalState,
        ) -> str:
        """
        Generate a solution step, return the parsed tactic and other information.
        """
        # Note: forward reasoning state should be re-formatted to solution state (case h.mp and ⊢ False)

    async def reset_async(self):
        """
        Clean garbabge
        """
        await logger.complete()
    
    async def search_async(
            self,
            solving_server: PropSolvingServer,
            init_forward_state: GoalState,
            init_solution_state: GoalState,
            tag: str='',
            verbose: bool=False,
        ) -> SolutionAutoregressionResult:
        """
        Solution autoregression from `init_state`
        """
        # Initialize
        assert solving_server.server.is_automatic(), "Search must be run in automatic mode"
        assert [(g.name, g.target) for g in init_forward_state.goals] == [(None, 'False')], 'Invalid init_forward_state'
        assert [g.name for g in init_solution_state.goals] == [g.name for g in init_solution_state.goals] == ['h.mp', 'h.mpr', 'w'], 'Invalid init_solution_state'
        
        time_start = time.time()
        solution = []
        states = [GoalState(
            state_id=-1,
            goals=init_forward_state.goals,
            payload={},
            _sentinel=[]
        )]
        submission = None
        backward_proof = None
        rpe_proof = None
        cur_forward_state = init_forward_state

        log = logger.info if verbose else logger.debug
        # Search
        try:
            i_trial = 0
            while i_trial < self.max_search_trials:
                i_trial += 1
                assert [(g.name, g.target) for g in cur_forward_state.goals] == [(None, 'False')], 'Error: Strange cur_forward_state: ```' + json.dumps(cur_forward_state.serialize()) + '```'
                step = await self.gen_step_async(solving_server.sample, cur_forward_state)
                log(f'Search({tag}): {i_trial}/{self.max_search_trials}, Step ```{str(step)}```')

                # If submitted, validate and return
                last_line = step.splitlines()[-1]
                if last_line.startswith('exact') and last_line.endswith('-- submit'):   # Submission
                    step_w_proof = SolutionStep(step=step, proofs=[])
                    assert len(remove_comments(step).strip().splitlines()) == 1, f'Error: Strange submission step'

                    solution_draft = '\n'.join([s.step for s in solution] + [step])
                    try:
                        backward_proving_state = await solving_server.server.goal_tactic_async(init_solution_state, 0, '{\n' + solution_draft + '\n}')
                    except TacticFailure as e:
                        logger.warning(f'Search({tag}): {i_trial}/{self.max_search_trials}, solution validation failed due to {e}')
                        await self.reset_async()
                        return SolutionAutoregressionResult(
                            duration=time.time() - time_start,
                            success=(rpe_proof is not None),
                            solution=solution,
                            states=states,
                            submission=submission,
                            backward_proof=backward_proof,
                            rpe_proof=rpe_proof,
                            cost=i_trial
                        )
                    assert [g.name for g in backward_proving_state.goals] == ['h.mpr'], 'Error: Strange backward proving state: ```' + json.dumps(backward_proving_state.serialize()) + '```'
                    solution.append(step_w_proof)
                    states.append(GoalState(
                        state_id=-1,
                        goals=backward_proving_state.goals,
                        payload={},
                        _sentinel=[]
                    ))

                    # Get submission
                    submission = await solving_server.get_submission_async(backward_proving_state)
                    logger.info(f'Search({tag}): {i_trial}/{self.max_search_trials}, submission: {submission}')

                    # Backward Proving
                    if self.proof_searcher is not None:
                        search_result = await self.proof_searcher.search_async(
                            server=solving_server.server,
                            init_state=backward_proving_state,
                            tag=tag+f'-{i_trial}/{self.max_search_trials}-backward',
                        )
                        if not search_result.success:
                            logger.info(f'Search({tag}): {i_trial}/{self.max_search_trials}, backward proof search failed')
                        else:
                            logger.info(f'Search({tag}): {i_trial}/{self.max_search_trials}, backward proof search succeeded with {search_result.proof}')
                            backward_proof = search_result.proof

                    # Check RPE
                    if solving_server.sample.formal_answer is not None:
                        try:
                            rpe_proof = await solving_server.prove_eq_async(submission)
                        except Exception as e:
                            logger.warning(f'Search({tag}): {i_trial}/{self.max_search_trials}: unexpected error in RPE Check for `{submission}`: {[traceback.format_exc()]}')
                    if rpe_proof is None:
                        logger.info(f'Search({tag}): {i_trial}/{self.max_search_trials}, RPE failed')
                    else:
                        logger.info(f'Search({tag}): {i_trial}/{self.max_search_trials}, RPE succeeded with `{rpe_proof}`')

                    log(f'Search({tag}): {i_trial}/{self.max_search_trials}, executed successfully.')

                    await self.reset_async()
                    return SolutionAutoregressionResult(
                        duration=time.time() - time_start,
                        success=(rpe_proof is not None),
                        solution=solution,
                        states=states,
                        submission=submission,
                        backward_proof=backward_proof,
                        rpe_proof=rpe_proof,
                        cost=i_trial
                    )
                else:
                    # Otherwise, process new search state and continue search
                    # Assuming all steps are correctly indented.
                    try:
                        next_forward_state = await solving_server.server.goal_tactic_async(cur_forward_state, 0, TacticDraft('by\n' + step + '\nsorry'))
                        assert next_forward_state.goals[-1].name is None and next_forward_state.goals[-1].target == 'False', 'Error: Strange last goal: ```' + to_simple_str(next_forward_state) + '```'
                    except TacticFailure as e:
                        logger.debug(f'Search({tag}): {i_trial}/{self.max_search_trials}, step drafting failed due to {e}')
                        continue

                    next_forward_goal = to_simple_str(next_forward_state.goals[-1])
                    # Proof search for logical gaps
                    if len(next_forward_state.goals) > 1:  # == 1: No gaps
                        if self.proof_searcher is None:
                            continue
                        search_result = await self.proof_searcher.search_async(
                            server=solving_server.server,
                            init_state=next_forward_state,
                            tag=tag+f'-{i_trial}/{self.max_search_trials}',
                            ignored_goals=[next_forward_state.goals[-1]]
                        )
                        if not search_result.success:
                            log(f'Search({tag}): {i_trial}/{self.max_search_trials}, proof search failed')
                            continue
                        else:
                            log(f'Search({tag}): {i_trial}/{self.max_search_trials}, proof search succeeded')
                        next_proven_forward_state = search_result.final_state
                        step_w_proof = SolutionStep(
                            step=step,
                            proofs=search_result.proof
                        )
                    else:
                        next_proven_forward_state = next_forward_state
                        step_w_proof = SolutionStep(
                            step=step,
                            proofs=[]
                        )
                    
                    assert next_forward_goal == to_simple_str(next_proven_forward_state), f'Error: Strange next_search_state: ```' + json.dumps({
                        'next_forward_state' : next_forward_state.serialize(),
                        'step' : step,
                        'next_proven_forward_state' : next_proven_forward_state.serialize()
                    }) + '```'

                    cur_forward_state = next_proven_forward_state
                    solution.append(step_w_proof)
                    states.append(GoalState(
                        state_id=-1,
                        goals=cur_forward_state.goals,
                        payload={},
                        _sentinel=[]
                    ))
                    log(f'Search({tag}): {i_trial}/{self.max_search_trials}, step executed successfully.')
        except Exception as e:
            logger.error(f'Search({tag}): {i_trial}/{self.max_search_trials}, fatal error```{[traceback.format_exc()]}```')

        logger.info(f'Search({tag}): search finished with {i_trial} expansions.')
        await self.reset_async()

        return SolutionAutoregressionResult(
            duration=time.time() - time_start,
            success=False,
            solution=solution,
            states=states,
            submission=submission,
            backward_proof=backward_proof,
            rpe_proof=rpe_proof,
            cost=i_trial
        )

class LLMSolutionAutoregressionAgent(SolutionAutoregressionAgent):
    """
    A template solution autoregression agent for LLM-based solution autoregression.
    Multiple search agent (each for one proposition) + one AsyncOpenAI-style API server
    """

    def __init__(self, gen_client: AsyncOpenAI, gen_model_name: str, proof_searcher: ProofSearchAgent, *args, max_search_trials: int=100, num_samples_per_trial: int=32, temperature: Optional[float]=None, max_tokens: int=2048, **kwargs) -> None:
        super().__init__(max_search_trials, proof_searcher)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')

        self.gen_client = gen_client
        self.gen_model_name = gen_model_name
        self.num_samples_per_trial = num_samples_per_trial
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def gen_prompt(self, sample: FormalProblem, cur_search_state: GoalState) -> List[Dict[str, str]]:
        """
        Generate a prompt for the generator.
        """

    async def gen_step_async(
            self,
            sample: FormalProblem,
            state: GoalState,
        ) -> str:
        """
        Given a GoalState, try at most `self.num_samples_per_trial` times to generate one step.
        """
        # Generate tactics
        for _ in range(self.num_samples_per_trial):
            try:
                if 'internlm' in self.gen_model_name.lower():
                    outputs = (await self.gen_client.chat.completions.create(
                        model=self.gen_model_name,
                        messages=self.gen_prompt(sample, state),
                        max_tokens=(self.max_tokens if (self.max_tokens != NOT_GIVEN and self.max_tokens > 0) else NOT_GIVEN),
                        stream=False,
                        temperature=self.temperature,
                        n=1,
                        stop='<|im_end|>'
                    )).choices
                else:
                    outputs = (await self.gen_client.chat.completions.create(
                        model=self.gen_model_name,
                        messages=self.gen_prompt(sample, state),
                        max_tokens=(self.max_tokens if (self.max_tokens != NOT_GIVEN and self.max_tokens > 0) else NOT_GIVEN),
                        stream=False,
                        temperature=self.temperature,
                        n=1,
                    )).choices
            except:
                logger.debug(f'Failed to generate tactics:\n{traceback.format_exc()}')
                continue
            
            # Neglect failed generations
            if not outputs[0].finish_reason == 'stop':
                logger.debug(f'gen_steps_async(): Tactic rejected due to abnormal finishing: {outputs[0].finish_reason}')
                continue
            
            step = extract_code(outputs[0].message.content)
            if len(step.strip()) == 0:
                continue

            if any(banned_token in remove_comments(str(step)) for banned_token in BANNED_TOKENS[1:]):   # Assuming the first banned token is `sorry`
                logger.warning(f'gen_steps_async(): Tactic `{remove_comments(str(step))}` rejected due to bannded token.')
                continue
            return step
        raise RuntimeError('LLM calling budget exceeded')

class SFT_NALP_LLMSolutionAutoregressionAgent(LLMSolutionAutoregressionAgent):
    """
    A solution autoregression agent with SFTed LLM.
    """
    def __init__(self, gen_client: AsyncOpenAI, gen_model_name: str, proof_searcher: ProofSearchAgent, *args, max_search_trials: int=100, num_samples_per_trial: int=32, temperature: Optional[float]=None, max_tokens: int=256, **kwargs) -> None:
        super().__init__(gen_client, gen_model_name, proof_searcher, max_search_trials=max_search_trials, num_samples_per_trial=num_samples_per_trial, temperature=temperature, max_tokens=max_tokens)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')

    def gen_prompt(self, sample: FormalProblem, cur_search_state: GoalState) -> List[Dict[str, str]]:
        # Assuming a valid forward reasoning state (initialized by `init_forward_reasoning_state_async`) with anonymous goal name and `False` target.
        solution_goal = 'case h.mp\n' + str(cur_search_state.goals[0])
        assert solution_goal.endswith('⊢ False')
        solution_goal = solution_goal[:-len('⊢ False')] + '⊢ ?w'
        return [
            {
                "role": "system",
                "content": "You are a Lean 4 expert."
            },
            {
                "role": "user",
                "content": format_forward_solution_step_prompt(informal_problem=sample.informal_problem, solution_goal=solution_goal)
            }
        ]
