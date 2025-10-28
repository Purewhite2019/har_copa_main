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
class SolutionSearchState:
    state: GoalState
    cost: float

    def __lt__(self, other: 'SolutionSearchState') -> bool:
        assert isinstance(other, SolutionSearchState)
        return self.cost < other.cost

@dataclass(frozen=True)
class SolutionStep:
    step: str
    proofs: List[str]

    def serialize(self) -> Dict:
        return {
            'step': self.step,
            'proofs': self.proofs
        } 

@dataclass(frozen=True)
class SolutionSearchResult:
    duration: float
    final_nodes: List[str]
    search_graph: nx.MultiDiGraph
    cost: int=float('inf')

    def serialize(self) -> Dict:
        return {
            'duration': self.duration,
            'final_nodes': self.final_nodes,
            'search_graph': nx.readwrite.node_link_data(self.search_graph),
            'cost': self.cost
        }

def to_simple_str(state: Union[Goal, GoalState, SolutionSearchState]) -> str:
    if isinstance(state, Goal):
        return json.dumps([state.serialize()])
    elif isinstance(state, GoalState):
        return json.dumps([g.serialize() for g in state.goals])
    elif isinstance(state, SolutionSearchState):
        return json.dumps([g.serialize() for g in state.state.goals])
    else:
        raise NotImplementedError(f'to_simple_str() is not implemented for {type(state)}.')

class SolutionSearchAgent:
    """
    A template best-first search agent for solution search. (The heuristic hasn't been implemented)
    """

    def __init__(self, max_search_trials: int, proof_searcher: Optional[ProofSearchAgent]) -> None:
        self.max_search_trials = max_search_trials
        self.proof_searcher = proof_searcher

    @abstractmethod
    async def gen_steps_async(
            self,
            sample: FormalProblem,
            state: SolutionSearchState,
        ) -> Tuple[List[Tactic], List[Any]]:
        """
        Generate several solution steps, return the parsed tactics and other information.
        """
        # Note: forward reasoning state should be re-formatted to solution state (case h.mp and ⊢ False)

    async def reset_async(self):
        """
        Clean garbabge
        """
        await logger.complete()

    @abstractmethod
    async def guidance_async(self, cur_search_state: SolutionSearchState, new_goal_state: GoalState, tactic: Tactic, tac_info: Any) -> float:
        """
        Return a priority determining which state should be searched
        first.
        """
    
    async def search_async(
            self,
            solving_server: PropSolvingServer,
            init_forward_state: GoalState,
            init_solution_state: GoalState,
            tag: str='',
            early_exit_with_submission: bool=True,
            verbose: bool=False,
        ) -> SolutionSearchResult:
        """
        Best-first solution search from `init_state`
        """
        # Initialize
        assert solving_server.server.is_automatic(), "Search must be run in automatic mode"
        assert [(g.name, g.target) for g in init_forward_state.goals] == [(None, 'False')], 'Invalid init_forward_state'
        assert [g.name for g in init_solution_state.goals] == [g.name for g in init_solution_state.goals] == ['h.mp', 'h.mpr', 'w'], 'Invalid init_solution_state'
        
        time_start = time.time()
        search_graph = nx.MultiDiGraph()
        # Node: (Serialized) Goals in GoalState (removed redundant information such as state_id and sentinel) 
        # Edge: Solution Step (tactic block with sorries and corresponding proofs)

        init_search_node = to_simple_str(init_forward_state)
        search_graph.add_node(init_search_node)
        search_pqueue: List[SolutionSearchState] = [
            SolutionSearchState(
                state=init_forward_state,
                cost=0.0
            )
        ]
        final_nodes = []
        log = logger.info if verbose else logger.debug
        # Search
        try:
            i_trial = 0
            while i_trial < self.max_search_trials:
                if len(search_pqueue) == 0:
                    logger.info(f'Search({tag}): {i_trial}/{self.max_search_trials}, len(S) = 0, exiting.')
                    break
                else:
                    log(f'Search({tag}): {i_trial}/{self.max_search_trials}, len(S) = {len(search_pqueue)}')
                
                cur_search_state = heapq.heappop(search_pqueue)
                cur_search_node = to_simple_str(cur_search_state)
                assert isinstance(cur_search_state, SolutionSearchState) and [(g.name, g.target) for g in cur_search_state.state.goals] == [(None, 'False')], 'Error: Strange cur_search_state: ```' + json.dumps(cur_search_state.state.serialize()) + '```'
                steps, step_infos = await self.gen_steps_async(solving_server.sample, cur_search_state)

                for i_step, (step, step_info) in enumerate(zip_strict(steps, step_infos)):
                    i_trial += 1
                    if i_trial >= self.max_search_trials:
                        break
                    log(f'Search({tag}): {i_trial}/{self.max_search_trials}, {i_step}/{len(steps)}, Step ```{str(step)}```')
                    
                    # If submitted, validate and return
                    last_line = step.splitlines()[-1]
                    if last_line.startswith('exact') and last_line.endswith('-- submit'):   # Submission
                        step_w_proof = SolutionStep(step=step, proofs=[])
                        assert len(remove_comments(step).strip().splitlines()) == 1, f'Error: Strange submission step'

                        edge_data = search_graph.edges
                        solution = list(reversed(
                            [edge_data[state_before, state_after, step_id]['step_w_proof']['step'] for state_after, state_before, step_id in nx.edge_dfs(search_graph.reverse(False), source=cur_search_node)]
                        )) + [step]

                        try:
                            backward_proving_state = await solving_server.server.goal_tactic_async(init_solution_state, 0, '{\n' + '\n'.join(solution) + '\n}')
                        except TacticFailure as e:
                            logger.warning(f'Search({tag}): {i_trial}/{self.max_search_trials}, solution validation failed due to {e}')
                            continue

                        assert [g.name for g in backward_proving_state.goals] == ['h.mpr'], 'Error: Strange backward proving state: ```' + json.dumps(backward_proving_state.serialize()) + '```'
                        backward_proving_node = to_simple_str(backward_proving_state)
                        if backward_proving_node in search_graph.nodes: # Existing node: skip post-validation
                            if all(d['step_w_proof'] != step_w_proof for u, v, d in search_graph.edges([cur_search_node, backward_proving_node], data=True)):
                                search_graph.add_edge(cur_search_node, backward_proving_node, step_w_proof=step_w_proof.serialize())
                            log(f'Search({tag}): {i_trial}/{self.max_search_trials}, {i_step}/{len(steps)}, skipped because backward_proving_node is also visited')
                            continue

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
                        else:
                            search_result = ProofSearchResult(
                                duration=0,
                                success=False,
                            )

                        # Check RPE
                        rpe_proof = None
                        if solving_server.sample.formal_answer is not None:
                            try:
                                rpe_proof = await solving_server.prove_eq_async(submission)
                            except Exception as e:
                                logger.warning(f'Search({tag}): {i_trial}/{self.max_search_trials}: unexpected error in RPE Check for `{submission}`: {[traceback.format_exc()]}')
                        if rpe_proof is None:
                            logger.info(f'Search({tag}): {i_trial}/{self.max_search_trials}, RPE failed')
                        else:
                            logger.info(f'Search({tag}): {i_trial}/{self.max_search_trials}, RPE succeeded with `{rpe_proof}`')

                        final_nodes.append(backward_proving_node)
                        search_graph.add_node(backward_proving_node, submission=submission, rpe_proof=rpe_proof, backward_proof=search_result.proof)
                        search_graph.add_edge(cur_search_node, backward_proving_node, step_w_proof=step_w_proof.serialize())
                        log(f'Search({tag}): {i_trial}/{self.max_search_trials}, executed successfully.')

                        if early_exit_with_submission or rpe_proof is not None:
                            await self.reset_async()
                            return SolutionSearchResult(
                                duration=time.time() - time_start,
                                final_nodes=final_nodes,
                                search_graph=search_graph,
                                cost=i_trial
                            )
                    else:
                        # Otherwise, process new search state and continue search
                        # Assuming all steps are correctly indented.
                        try:
                            next_goal_state = await solving_server.server.goal_tactic_async(cur_search_state.state, 0, TacticDraft('by\n' + step + '\nsorry'))
                            assert next_goal_state.goals[-1].name is None and next_goal_state.goals[-1].target == 'False', 'Error: Strange last goal: ```' + json.dumps(next_goal_state.state.serialize()) + '```'
                        except TacticFailure as e:
                            logger.debug(f'Search({tag}): {i_trial}/{self.max_search_trials}, step drafting failed due to {e}')
                            continue

                        next_search_node = to_simple_str(next_goal_state.goals[-1])
                        if next_search_node in search_graph.nodes:  # If visited, skip
                            log(f'Search({tag}): {i_trial}/{self.max_search_trials}, {i_step}/{len(steps)}, next_search_node in search_graph, skipped.')
                            continue

                        # Proof search for logical gaps
                        if len(next_goal_state.goals) > 1:  # == 1: No gaps
                            if self.proof_searcher is None:
                                continue
                            search_result = await self.proof_searcher.search_async(
                                server=solving_server.server,
                                init_state=next_goal_state,
                                tag=tag+f'-{i_trial}/{self.max_search_trials}',
                                ignored_goals=[next_goal_state.goals[-1]]
                            )
                            if not search_result.success:
                                log(f'Search({tag}): {i_trial}/{self.max_search_trials}, {i_step}/{len(steps)}, proof search failed')
                                continue
                            else:
                                log(f'Search({tag}): {i_trial}/{self.max_search_trials}, {i_step}/{len(steps)}, proof search succeeded')
                            next_goal_state = search_result.final_state
                            step_w_proof = SolutionStep(
                                step=step,
                                proofs=search_result.proof
                            )
                        else:
                            step_w_proof = SolutionStep(
                                step=step,
                                proofs=[]
                            )
                        
                        assert next_search_node == to_simple_str(next_goal_state), f'Error: Strange next_search_state: ```' + json.dumps({
                            'cur_goal_state' : cur_search_state.state.serialize(),
                            'step' : step,
                            'next_goal_state' : next_goal_state.state.serialize()
                        }) + '```'
                        
                        next_search_state = SolutionSearchState(
                            state=next_goal_state,
                            cost=await self.guidance_async(cur_search_state, next_goal_state, step, step_info)
                        )
                        search_graph.add_node(next_search_node) # Add node
                        search_graph.add_edge(cur_search_node, next_search_node, step_w_proof=step_w_proof.serialize()) # Add edge
                        heapq.heappush(search_pqueue, next_search_state)
                        log(f'Search({tag}): {i_trial}/{self.max_search_trials}, step executed successfully.')
        except Exception as e:
            logger.error(f'Search({tag}): {i_trial}/{self.max_search_trials}, fatal error```{[traceback.format_exc()]}```')

        logger.info(f'Search({tag}): search finished with {i_trial} expansions.')
        await self.reset_async()

        return SolutionSearchResult(
            duration=time.time() - time_start,
            final_nodes=final_nodes,
            search_graph=search_graph,
            cost=i_trial
        )

class LLMSolutionSearchAgent(SolutionSearchAgent):
    """
    A template best-first search agent for LLM-based solution search. (The heuristic hasn't been implemented)
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
    def gen_prompt(self, sample: FormalProblem, cur_search_state: SolutionSearchState) -> List[Dict[str, str]]:
        """
        Generate a prompt for the generator.
        """

    async def gen_steps_async(
            self,
            sample: FormalProblem,
            state: SolutionSearchState,
        ) -> Tuple[List[Tactic], List[Choice]]:
        """
        Given a GoalState, generate `self.num_samples_per_trial` steps, return parsed tactics and scores.
        """
        generated: Set[str] = set()
        steps, step_infos = [], []
        outputs: List[Choice] = []

        # Generate tactics
        try:
            if self.gen_model_name in ['deepseek-chat', 'deepseek-coder']:
                outputs = [r.choices[0] for r in asyncio.gather(*[
                await self.gen_client.chat.completions.create(
                    model=self.gen_model_name,
                    messages=self.gen_prompt(sample, state),
                    max_tokens=(self.max_tokens if (self.max_tokens != NOT_GIVEN and self.max_tokens > 0) else NOT_GIVEN),
                    stream=False,
                    temperature=self.temperature,
                    logprobs=True,
                )
                for _ in range(self.num_samples_per_trial)
            ])] # Deepseek API doesn't support `n`
            elif 'internlm' in self.gen_model_name.lower():
                outputs = (await self.gen_client.chat.completions.create(
                    model=self.gen_model_name,
                    messages=self.gen_prompt(sample, state),
                    max_tokens=(self.max_tokens if (self.max_tokens != NOT_GIVEN and self.max_tokens > 0) else NOT_GIVEN),
                    stream=False,
                    temperature=self.temperature,
                    n=self.num_samples_per_trial,
                    logprobs=True,
                    stop='<|im_end|>'
                )).choices
            else:
                outputs = (await self.gen_client.chat.completions.create(
                    model=self.gen_model_name,
                    messages=self.gen_prompt(sample, state),
                    max_tokens=(self.max_tokens if (self.max_tokens != NOT_GIVEN and self.max_tokens > 0) else NOT_GIVEN),
                    stream=False,
                    temperature=self.temperature,
                    n=self.num_samples_per_trial,
                    logprobs=True,
                )).choices
        except:
            logger.error(f'Failed to generate tactics:\n{traceback.format_exc()}')
        
        for output in outputs:
            # Neglect failed generations
            if not output.finish_reason == 'stop':
                logger.debug(f'gen_steps_async(): Tactic rejected due to abnormal finishing: {output.finish_reason}')
                continue
            
            step = extract_code(output.message.content)
            if len(step.strip()) == 0:
                continue
            normalized_step = normalize_spaces(remove_comments(step))

            if normalized_step not in generated:   # Remove replicates
                generated.add(normalized_step)
                if any(banned_token in remove_comments(str(step)) for banned_token in BANNED_TOKENS[1:]):   # Assuming the first banned token is `sorry`
                    logger.warning(f'gen_steps_async(): Tactic `{remove_comments(str(step))}` rejected due to bannded token.')
                    continue
                steps.append(step)
                step_infos.append(output)
        return steps, step_infos

class SFT_NALP_LLMSolutionSearchAgent(LLMSolutionSearchAgent):
    """
    A best-first search agent with SFTed LLM and normalized accumulated tactic log-probabilities as heuristic.
    """
    def __init__(self, gen_client: AsyncOpenAI, gen_model_name: str, proof_searcher: ProofSearchAgent, *args, max_search_trials: int=100, num_samples_per_trial: int=32, temperature: Optional[float]=None, max_tokens: int=256, **kwargs) -> None:
        super().__init__(gen_client, gen_model_name, proof_searcher, max_search_trials=max_search_trials, num_samples_per_trial=num_samples_per_trial, temperature=temperature, max_tokens=max_tokens)
        if len(args) > 0 or len(kwargs) > 0:
            logger.warning(f'Redundant arguments for {type(self)}: {args} {kwargs}')

    def gen_prompt(self, sample: FormalProblem, cur_search_state: SolutionSearchState) -> List[Dict[str, str]]:
        # Assuming a valid forward reasoning state (initialized by `init_forward_reasoning_state_async`) with anonymous goal name and `False` target.
        solution_goal = 'case h.mp\n' + str(cur_search_state.state.goals[0])
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

    async def guidance_async(self, cur_search_state: SolutionSearchState, new_goal_state: GoalState, tactic: Tactic, tac_info: Choice) -> float:
        return cur_search_state.cost + sum([-o.logprob for o in tac_info.logprobs.content]) / max(len(tac_info.logprobs.content), 1)
