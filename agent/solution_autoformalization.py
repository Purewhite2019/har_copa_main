import os.path as osp
from abc import abstractmethod
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Dict, Set, Any, Callable, Coroutine
import collections as C
import itertools as I
import heapq
import asyncio
import traceback
import regex as re
import warnings

from loguru import logger
import pexpect
# logger = logger.opt(colors=True)  # Some tactics or proof states might contain tokens like <...>, </...> that results in error in loguru.
from openai import OpenAI, AsyncOpenAI, NOT_GIVEN
from openai.types.chat.chat_completion import Choice
from easydict import EasyDict
import vllm
from transformers import AutoTokenizer

from common.constants import BANNED_TOKENS, NEW_LINE, ANSWER_PATTERN, CODEBLOCK_PATTERN, SYSTEM_PROMPT_SFT, OPEN_HEADER, H_SUBMISSION_NAME  # if 'hf_ckpts' not in self.model, use sft prompt
from common.pantograph.server import Server
from common.pantograph.solving_server import PropSolvingServer, format_variable_sequence
from common.pantograph.dataclasses import TacticHave, TacticLet, Tactic, GoalState, Goal, TokenUsage, DataPoint, FormalProblem, SolutionAutoformalizationResult, TacticDraft
from common.utils import zip_strict, remove_comments, to_sync, remove_spaces, replace_sorry
from agent.proof_search import ProofSearchResult, ProofSearchAgent


class SolutionAutoformalizer:    
    def __init__(
            self,
            server: Server,
            proof_searcher: Optional[ProofSearchAgent]=None,
            autoformalization_client: Optional[AsyncOpenAI]=None,
            autoformalization_model: str=None,
            demonstrations: Optional[Dict[str, Dict[str, List[DataPoint]]]]=None,
            try_num: int=1,
            temperature: float=1.0,
            dynamic_temperatre: bool=True,
            max_tokens: int=2048
        ):
        # demonstrations[subject][level][i] ~ 'informal_problem', 'informal_answer', 'informal_solution', 'level', 'subject', 'formal_problem', 'formal_answer', 'formal_solution_draft', 'annotator'
        self.server = server
        self.proof_searcher = proof_searcher
        self.autoformalization_client = autoformalization_client
        self.autoformalization_model = autoformalization_model
        self.demonstrations = demonstrations if demonstrations is not None else {}
        self.try_num = try_num
        self.temperature = temperature
        self.dynamic_temperatre = dynamic_temperatre
        self.max_tokens = max_tokens
    
    async def autoformalize_async(
        self, problem_data: Union[DataPoint, Dict[str, str]],
        tag: Optional[str]=None,
    ) -> SolutionAutoformalizationResult:
        logger.debug(f'autoformalize_async({tag}): started.')
        state = SolutionAutoformalizationResult.from_kwargs(
            **problem_data.__dict__
        )
        assert state.formal_statement is not None
        if 'proof_search_results' not in state.metainfo.keys():
            state.metainfo['proof_search_results'] = []
        
        # Metainfo update
        if 'annotator' not in state.metainfo:
            state.metainfo['annotator'] = list()
        if self.autoformalization_model not in state.metainfo['annotator']:
            state.metainfo['annotator'].append(self.autoformalization_model)
        if tag is None and hasattr(problem_data, 'subject') and hasattr(problem_data, 'level'):
            tag = problem_data.subject + '-' + problem_data.level

        # Hierarchical Solution Autoformalization
        units = await self.server.load_sorry_async(
            state.header or OPEN_HEADER + '\n' + state.formal_statement
        )
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), str([x.messages for x in units])
        init_proof_state = units[-1].goal_state
        if self.autoformalization_client is None:
            raise ValueError("'self.client' shouldn't be None for autoformalization")
        prompt = self.format_solution_draft_prompt(state)
        for try_i in range(self.try_num):
            try:
                # Solution Draft ≈ Proof Draft
                response = (await self.autoformalization_client.chat.completions.create(
                    model=self.autoformalization_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant." if 'hf_ckpts' not in self.autoformalization_model else SYSTEM_PROMPT_SFT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=(self.max_tokens if (self.max_tokens != NOT_GIVEN and self.max_tokens > 0) else NOT_GIVEN),
                    stream=False,
                    temperature=self.temperature if not self.dynamic_temperatre else (self.temperature / self.try_num * try_i),
                    stop=["<|im_end|>"] if 'internlm' in self.autoformalization_model else NOT_GIVEN
                ))
                state.token_usages['solution_autoformalization'] += response.usage
                if response.choices[0].finish_reason != 'stop':
                    logger.warning(f'autoformalize_async.solution_autoformalization({tag}, {try_i}/{self.try_num}): Invalid request finish reason: {response.choices[0].finish_reason}')
                parse_result = re.findall(CODEBLOCK_PATTERN, response.choices[0].message.content)
                assert len(parse_result) == 1, f"response: {response.choices[0].message.content}"

                # 1. Ensure all 'by sorry' in formal_solution_draft to be 'sorry'
                formal_solution_draft = replace_sorry(parse_result[0].strip())
                # 2. Deductive Check
                lines = formal_solution_draft.rstrip().split('\n')
                submission_line = lines[-1]
                assert submission_line.startswith('exact ') and submission_line.endswith('-- submit'), f'submission_line: {submission_line}'
                
                # Typecheck
                final_proof_state = await self.server.goal_tactic_async(init_proof_state, 0, '{\n' + formal_solution_draft + '\n}')
                assert final_proof_state.is_solved
                
                # Change goal to False to ensure the tactics haven't changed the goal (such that submission is exactly (def. eq. to) the answer)
                solution_wo_submission = '\n'.join(lines[:-1]) + f'\nsorry'
                false_solution_state = await self.server.goal_tactic_async(init_proof_state, 0, 'exfalso')
                final_proof_state = await self.server.goal_tactic_async(false_solution_state, 0, '{\n' + solution_wo_submission + '\n}')
                assert final_proof_state.is_solved
                
                # Succeed
                logger.opt(colors=True).info(f'<green>autoformalize_async({tag}): Solution autoformalization succeeded.</green>')
                state.formal_solution_draft = formal_solution_draft
                break
            except Exception as e:
                logger.debug(f'autoformalize_async.solution_autoformalization({tag}, {try_i}/{self.try_num}): Failed with {e}, retrying...')
        # All Fail
        if state.formal_solution_draft is None:
            logger.warning(f'autoformalize_async({tag}): Solution autoformalization failed in all attempts.')
            return state

        # Solution Draft + Gap Proofs = Solution
        try:
            lines = formal_solution_draft.rstrip().split('\n')
            submission_line = lines[-1]
            assert submission_line.startswith('exact ') and submission_line.endswith('-- submit'), f'submission_line: {submission_line}'

            sorry_state = await self.server.goal_tactic_async(init_proof_state, 0, TacticDraft('by {\n' + state.formal_solution_draft + '\n}'))
        except Exception as e:
            logger.error(traceback.format_exc())
            import pdb; pdb.set_trace()
            print(e)

        # Separately solve each sorry by initializing them into new search states        
        formal_gaps = sorry_state.goals[:]
        state.formal_proofs = []
        while sorry_state is not None and not sorry_state.is_solved:
            logger.info(f'autoformalize_async.proof_search({tag}): Searching proof {len(formal_gaps)-len(sorry_state.goals)}/{len(formal_gaps)}...')
            try:
                search_result = await self.proof_searcher.search_async(
                    server=self.server,
                    init_state=sorry_state,
                    tag=tag+f'-{len(formal_gaps)-len(sorry_state.goals)}',
                    ignored_goals=sorry_state.goals[1:]
                )
            except Exception as e:
                logger.error(f'autoformalize_async({tag}): Proof search failed: Exception {e}:\n{traceback.format_exc()}.')
                if len(formal_gaps) == len(state.formal_proofs):
                    logger.critical(f'autoformalize_async({tag}): len(formal_gaps) == len(state.formal_proofs) but fails.')
                state.metainfo['proof_search_results'].extend([{   # To save more data
                    'category': 'proof_search',
                    'result': r.serialize()
                } for r in state.formal_proofs])
                state.formal_proofs = []
                return state
            # If not proven, early exit
            if not search_result.success:
                logger.warning(f'autoformalize_async({tag}): Proof search failed.')
                if len(formal_gaps) == len(state.formal_proofs):
                    logger.critical(f'autoformalize_async({tag}): len(formal_gaps) == len(state.formal_proofs) but fails.')
                state.metainfo['proof_search_results'].extend([{   # To save more data
                    'category': 'proof_search',
                    'result': r.serialize()
                } for r in state.formal_proofs])
                state.formal_proofs = []  # To save more data
                return state
            state.formal_proofs.append(search_result)
            sorry_state = search_result.final_state
        if sorry_state is None or sorry_state.is_solved:
            logger.opt(colors=True).info(f'<green>autoformalize_async({tag}): Proof search succeeded.</green>')
            # All sorries are successfully proven 
            state.success = True
            if len(formal_gaps) != len(state.formal_proofs):
                logger.critical(f'autoformalize_async({tag}): len(formal_gaps) != len(state.formal_proofs) but succeeds.')
        else:
            assert sorry_state is None
            logger.opt(colors=True).info(f'<yellow>autoformalize_async({tag}): Proof search failed.</yellow>')
        return state
    
    autoformalize = to_sync(autoformalize_async)

    def format_informal_solving_prompt(self, v: Union[DataPoint, FormalProblem]) -> str:
        return self.informal_solving_demonstrations + '\n\n' + f'''
Question: {v.informal_problem}
Answer: Let's think step by step
'''

    def format_sample(self, v: Union[DataPoint, FormalProblem], stop_at_proof: bool=False) -> str:
        '''
        v : 'informal_problem', 'informal_answer', 'informal_solution',
            'level', 'subject',
            'formal_problem', 'formal_answer', 'formal_solution_draft',
            'annotator'
        '''
        text = f'''# Problem
"""
{v.informal_problem}
"""
# Answer
"""
the answer is {v.informal_answer}
"""
'''
        if not stop_at_proof:
            text += f'''# Solution
"""
{v.informal_solution}
"""
'''
        if isinstance(v, DataPoint):
            if v.formal_problem is not None:
                assert v.formal_answer is not None
                assert v.formal_answer_type is not None
                text += f'''
# Formal Statement
```lean4
example (answer : {v.formal_answer_type})
  {v.formal_problem}
-- Goal: {v.informal_answer}
: {v.formal_answer} := by
```
'''
        elif isinstance(v, FormalProblem):
            if v.formal_statement is not None:
                assert v.formal_statement.endswith('\n:= sorry')
                text += f'''
# Formal Statement
```lean4
{v.formal_statement[:-len('sorry')]}by
```
'''
        else:
            raise TypeError('v should be either DataPoint or FormalProblem')
        if v.formal_solution_draft is not None and not stop_at_proof:
            text += f'''
# Formal Proof Draft
```lean4
{v.formal_solution_draft}
```
'''
        return text

    def format_problem_autoformalization_prompt(self, sample: Union[DataPoint, FormalProblem]) -> str:
        '''
        sample : 'informal_problem', 'informal_answer', 'informal_solution',
                    'level', 'subject',
                    'formal_problem', 'formal_answer', 'formal_solution_draft',
                    'annotator'
        Input: 'informal_problem', 'informal_answer', 'informal_solution'
        Output: 'formal_problem + answer'
        '''
        if isinstance(sample, DataPoint):
            assert sample.formal_problem is None and sample.formal_answer is None
        else:
            assert sample.formal_statement is None

        prompt = f'''Given a natural language math problem and its answer, please generate a corresponding Lean 4 formal statement.
Please add comments highlighting the original parts of the natural language math problem and answer.
Please explicitly use the variable `answer` to indicate the answer in the formal statement.
Please maintain the semantic equivalence between natural language math and Lean 4.
Please assume the following header code has already been executed and do not add any imports or openings.
```lean4
import Mathlib
{OPEN_HEADER}
```

'''
        subject = sample.subject if isinstance(sample, DataPoint) else sample.metainfo.get('subject', None)
        level = sample.level if isinstance(sample, DataPoint) else sample.metainfo.get('level', None)
        if subject in self.demonstrations.keys() and level in self.demonstrations[subject].keys():
            prompt += 'Here are some examples:\n\n'
            for v in self.demonstrations[subject][level]:
                prompt += self.format_sample(v, stop_at_proof=True)
                prompt += '\n---\n\n'

            prompt += 'Now, please generate a formal statement for the following problem.\n\n'
            
        prompt += self.format_sample(sample, stop_at_proof=True) + '\n'
        return prompt
    
    def format_solution_draft_prompt(self, sample: Union[DataPoint, FormalProblem]) -> str:
        '''
        sample : 'informal_problem', 'informal_answer', 'informal_solution',
                    'level', 'subject',
                    'formal_problem', 'formal_answer', 'formal_solution_draft',
                    'annotator'
        Input: 'informal_problem', 'informal_answer', 'informal_solution', 'formal_problem', 'formal_answer'
        Output: 'formal_solution_draft'
        '''
        assert sample.formal_solution_draft is None
        prompt = '''Given a natural language math problem, its answer, its solution, and its formal statement, please generate a corresponding Lean 4 proof sketch and add comments to highlight the original parts of the natural language math solution.
Please maintain the semantic equivalence between natural language math and Lean 4.
Please only use forward reasoning in the proof, do not use tactics that modify the final goal.
For new hypotheses, please do not prove them and use `sorry` to close them.
Please assume the following header code has already been executed and do not add any imports or openings.
```lean4
import Mathlib
```

'''
        subject = sample.subject if isinstance(sample, DataPoint) else sample.metainfo.get('subject', None)
        level = sample.level if isinstance(sample, DataPoint) else sample.metainfo.get('level', None)
        if subject in self.demonstrations.keys() and level in self.demonstrations[subject].keys():
            prompt += 'Here are some examples:\n\n'
            for v in self.demonstrations[subject][level]:
                if v.formal_solution_draft is None:
                    continue
                prompt += self.format_sample(v)
                prompt += '\n---\n\n'

            prompt += 'Now, please generate a proof sketch for the following problem.\n\n'
        
        prompt += self.format_sample(sample) + '\n'
        return prompt
