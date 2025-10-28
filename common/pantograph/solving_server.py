
import collections as C
import functools as F
import itertools as I
from typing import Optional, Union, List, Dict, Tuple, Set, Iterable
from dataclasses import dataclass, field, fields
import regex as re
import traceback

from loguru import logger
import pexpect

from common.constants import CORE_OPTIONS, PARSING_EXTRA_OPTIONS, OPEN_HEADER, RPE_TACTICS, H_SUBMISSION_NAME, BANNED_TOKENS_IN_SOLVING_STATE, BANNED_TOKENS_IN_ANSWER_TYPE
from common.utils import (
    format_variable_sequence,
    parse_idents,
    replace_sorry,
    remove_comments,
    to_sync)
from common.pantograph.server import Server, TacticFailure, ServerError
from common.pantograph.dataclasses import GoalState, Variable, Goal, DataPoint, FormalProblem, TacticDraft

bracket_pairings = {
    '(' : ')',
    '[' : ']',
    '{' : '}',
    '⦃' : '⦄'
}

def parse_variables(s : str) -> str:
    base = 0
    extracted = []
    while base < len(s):
        if s[base] in ['(', '[', '{', '⦃']:
            bracket_type = s[base]
            bracket_pairing = bracket_pairings[bracket_type]
        
            stack_cnt = 0
            start_end_positions = []

            for i, char in enumerate(s[base:]):
                if char == bracket_type:
                    if stack_cnt == 0:
                        start_position = i
                    stack_cnt += 1
                elif char == bracket_pairing:
                    if stack_cnt > 0:
                        stack_cnt -= 1
                        if stack_cnt == 0:
                            end_position = i
                            start_end_positions.append((start_position, end_position))
                            break
            
            start, end = start_end_positions[0]
            extracted.append(s[base+start:base+end+1])
            base += i
        else:
            if s[base] == ':':
                break
            base += 1
    
    return extracted

class BaseSolvingServer:
    def __init__(
        self,
        imports: List[str]=["Mathlib", "Aesop"],
        project_path: Optional[str]=None,
        timeout: int=300,
        tag: str=''
    ) -> None:
        self.sample : Optional[FormalProblem] = None
        self.answer_mvarId : Optional[str] = None

        self.server: Optional[Server] = None
        self.imports = imports
        self.project_path = project_path
        self.timeout = timeout
        self.tag = tag

    async def init_backward_state_async(self, sample: Optional[Union[FormalProblem, DataPoint, str]]=None) -> GoalState:
        await self.load_problem_async(sample)
        
        self.formal_problem_framework = 'example : ∀ ' + \
            ((format_variable_sequence(self.sample.intros) + ' ') if len(self.sample.intros) > 0 else '') + \
            f'(answer : {self.sample.formal_answer_type}) ({H_SUBMISSION_NAME}: {self.sample.formal_answer}) ' + \
            ((format_variable_sequence(self.sample.outros[:-1]) + ' ') if len(self.sample.outros) > 1 else '') + \
            f', ({self.sample.outros[-1].t})\n:= sorry\n'

        units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER)+'\n'+self.formal_problem_framework)
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), f'formal_problem_framework:{(self.sample.header or OPEN_HEADER)}\n{self.formal_problem_framework}'
        return units[-1].goal_state
            
    init_backward_state = to_sync(init_backward_state_async)

    async def load_problem_async(self, sample: Optional[Union[FormalProblem, DataPoint, str]], force_parse: bool=False) -> None:
        # return None: loaded a parsed sample; Set[str]: loaded a not-previously-parsed sample, thus requires further validation
        if sample is None:
            return None
        
        if isinstance(sample, FormalProblem):
            self.sample = sample
            if not force_parse:
                # Assume problems in `FormalProblem` style pass all checks, such as type check, falsify check, etc.
                self.server = await Server.create(
                    imports=self.imports,
                    project_path=self.project_path,
                    core_options=CORE_OPTIONS,
                    timeout=300,
                )
                return None
        elif isinstance(sample, DataPoint):
            # DataPoint is a deprecated dataclass.
            self.sample = FormalProblem(
                informal_problem=sample.informal_problem,
                informal_answer=sample.informal_answer,
                informal_solution=sample.informal_solution,
                header=sample.header or OPEN_HEADER,
                formal_statement=f'example\n{sample.formal_problem}\n: {sample.formal_answer}\n:= sorry' if sample.formal_answer_type is None else f'example\n(answer : {sample.formal_answer_type})\n{sample.formal_problem}\n: {sample.formal_answer}\n:= sorry',
                formal_answer=sample.formal_answer,
                metainfo=dict(
                    level=sample.level,
                    subject=sample.subject,
                    annotator=sample.annotator
                )
            )
        else:
            assert isinstance(sample, str)
            self.sample = FormalProblem(
                header=OPEN_HEADER,
                formal_statement=replace_sorry(sample)
            )
        
        self.server = await Server.create(
            imports=self.imports,
            project_path=self.project_path,
            core_options=CORE_OPTIONS + PARSING_EXTRA_OPTIONS,
            timeout=300,
        )
        
        formal_statement = remove_comments((self.sample.header or OPEN_HEADER)+'\n'+self.sample.formal_statement)
        init_states = await self.server.load_sorry_async(formal_statement) # self.sample.header is added to help parse existing data
        assert len(init_states) >= 1 and 'error' not in str([x.messages for x in init_states]) and init_states[-1].goal_state is not None and len(init_states[-1].goal_state.goals) == 1, f'load_problem_async({self.tag}): formal_problem:\n{formal_statement}'
        
        init_state_str = str(init_states[-1].goal_state)
        assert H_SUBMISSION_NAME not in init_state_str, f'load_problem_async({self.tag}): Statement conflict with H_SUBMISSION_NAME "{H_SUBMISSION_NAME}" in {[init_state_str]}'
        for t in BANNED_TOKENS_IN_SOLVING_STATE:
            assert t not in init_state_str, f'load_problem_async({self.tag}): Detected banned token "{t}" in {[init_state_str]}'
        
        formal_problem_parsed = init_states[-1].goal_state.goals[0]
        formal_answer_components = set(parse_idents(formal_problem_parsed.target))
        
        self.sample.formal_answer = formal_problem_parsed.target
            
        if 'answer' not in remove_comments(self.sample.formal_answer).strip():
            logger.warning(f'load_problem_async({self.tag}): Invalid formal answer: {remove_comments(self.sample.formal_answer).strip()}, running in unsupervised mode.')
            self.sample.formal_answer = None

        intros = []
        outros = None
        
        named_vartypes = dict()
        anonymous_vartypes = []
        for line in parse_variables(formal_statement):
            if not line.startswith('['):
                line = line[1:-1]
                split_pos = line.find(':')
                assert split_pos != -1, f'load_problem_async({self.tag}): Unable to find variable names in `{line}`'
                var_names, var_type = line[:split_pos].strip(), line[split_pos+1:].strip()
                for var_name in var_names.split():
                    named_vartypes[var_name] = var_type
            else:
                anonymous_vartypes.append(line[1:-1].strip())
        
        
        for v in formal_problem_parsed.variables:
            assert v.t is not None and v.v is None and v.name is not None
            if '✝' in v.name:
                assert v.name.startswith('inst✝'), f"load_problem_async({self.tag}): '✝' detected in non-instance variable: {v.name}"
                v = Variable(
                    t=anonymous_vartypes.pop(0),
                    v=None,
                    name=None,
                )
            else:
                v = Variable(
                    t=named_vartypes.get(v.name, None) or v.t,  # Prefer parsed var types to contextual var types
                    v=None,
                    name=v.name,
                )                
            if v.name == 'answer':
                assert outros is None
                outros = []
                self.sample.formal_answer_type = v.t
                for t in BANNED_TOKENS_IN_ANSWER_TYPE:
                    assert t not in self.sample.formal_answer_type, f'load_problem_async({self.tag}): Detected banned token "{t}" in answer type "{self.sample.formal_answer_type}"'
                continue
            else:
                if outros is None:
                    intros.append(v)
                else:
                    # Identify which variable the `answer` depends and add them to `intro`
                    if v.name in formal_answer_components:
                        intros.append(v)
                    else:
                        outros.append(v)
        assert outros is not None and len(outros) > 0  # sometimes there might be comments in `answer`

        self.sample.intros = intros
        self.sample.outros = outros
        
        # Re-initialization
        extra_args = ' '.join(['--' + o for o in PARSING_EXTRA_OPTIONS])
        assert self.server.args.endswith(extra_args)
        self.server.args = self.server.args[:-len(extra_args)]
        await self.server.restart_async()
    
    load_problem = to_sync(load_problem_async)

    async def get_submission_async(self, state: GoalState) -> Optional[str]:
        assert self.answer_mvarId is not None
        rs = await self.server.goal_print_async(state, False, False, False, [self.answer_mvarId])
        answer = rs['extraMVars'][0]
        return answer.get('pp', None)

    get_submission = to_sync(get_submission_async)

# Answer as Prop
class PropSolvingServer(BaseSolvingServer):
    # Load from a Datapoint or statement, returning current solution state and solution goals to ignore
    async def init_solving_state_async(self, sample: Optional[Union[FormalProblem, DataPoint, str]]=None) -> GoalState:
        await self.load_problem_async(sample)
        
        self.answer_mvarId = None
        self.formal_problem_framework = 'example :' + \
            ((' ∀ ' + format_variable_sequence(self.sample.intros) + ',\n') if len(self.sample.intros) > 0 else '\n') + \
            f'∀ (answer : {self.sample.formal_answer_type}), ∃ ({H_SUBMISSION_NAME} : Prop), ' + \
            (('\n∀ ' + format_variable_sequence(self.sample.outros[:-1]) + ',\n') if len(self.sample.outros) > 1 else '\n') + \
            f'(({self.sample.outros[-1].t}) ↔ {H_SUBMISSION_NAME})\n:= sorry'
        
        units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER)+'\n'+self.formal_problem_framework)
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), f'formal_problem_framework:{(self.sample.header or OPEN_HEADER)}\n{self.formal_problem_framework}'
        init_solution_state = units[-1].goal_state
        
        if len(self.sample.intros) > 0:
            init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, 'intros ' + ' '.join([v.name or '_' for v in self.sample.intros]))
        init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, 'intros answer')
        init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, 'apply Exists.intro')
        
        rs = await self.server.goal_print_async(init_solution_state, False, False, True)
        answer_mvarId = [g['name'] for g in rs['goals'] if g['userName'] == 'w']
        assert len(answer_mvarId) == 1
        self.answer_mvarId = answer_mvarId[0]
        
        if len(self.sample.outros) > 1:
            init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, 'intros ' + ' '.join([v.name or '_' for v in self.sample.outros[:-1]]))
        init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, 'constructor')
        init_solution_state = await self.server.goal_tactic_async(init_solution_state, 1, f'intros {H_SUBMISSION_NAME}')
        init_solution_state = await self.server.goal_tactic_async(init_solution_state, 1, 'intros ' + self.sample.outros[-1].name)
        
        assert len(init_solution_state.goals) == 3 and [g.name for g in init_solution_state.goals] == ['h.mp', 'h.mpr', 'w'], f'Invalid solving_state:\n{init_solution_state}'
    
        return init_solution_state
        
    init_solving_state = to_sync(init_solving_state_async)

    async def init_forward_solving_state_async(self, sample: Optional[Union[FormalProblem, DataPoint, str]]=None) -> GoalState:
        await self.load_problem_async(sample)
        
        self.answer_mvarId = None
        self.formal_problem_framework = 'example :\n' + \
            (('∀ ' + format_variable_sequence(self.sample.intros) + ',\n') if len(self.sample.intros) > 0 else '') + \
            f'∀ (answer : {self.sample.formal_answer_type}), ∃ ({H_SUBMISSION_NAME} : Prop), \n' + \
            (('∀ ' + format_variable_sequence(self.sample.outros) + ',\n') if len(self.sample.outros) > 0 else '') + \
            f'({H_SUBMISSION_NAME})' + '\n:= sorry'
        
        units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER)+'\n'+self.formal_problem_framework)
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), f'formal_problem_framework:{(self.sample.header or OPEN_HEADER)}\n{self.formal_problem_framework}'
        init_solution_state = units[-1].goal_state
        
        if len(self.sample.intros) > 0:
            init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, 'intros ' + ' '.join([v.name or '_' for v in self.sample.intros]))
        init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, 'intros answer')
        init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, 'apply Exists.intro')
        
        rs = await self.server.goal_print_async(init_solution_state, False, False, True)
        answer_mvarId = [g['name'] for g in rs['goals'] if g['userName'] == 'w']
        assert len(answer_mvarId) == 1
        self.answer_mvarId = answer_mvarId[0]
        
        if len(self.sample.outros) > 0:
            init_solution_state = await self.server.goal_tactic_async(init_solution_state, 0, 'intros ' + ' '.join([v.name or '_' for v in self.sample.outros]))
        
        assert [g.name for g in init_solution_state.goals] == ['h', 'w'], f'Invalid solving_state:\n{init_solution_state}'
    
        return init_solution_state
        
    init_forward_solving_state = to_sync(init_forward_solving_state_async)

    async def init_forward_reasoning_state_async(self, sample: Optional[Union[FormalProblem, DataPoint, str]]=None) -> GoalState:
        await self.load_problem_async(sample)
        
        self.formal_problem_framework = 'example ' + \
            ((format_variable_sequence(self.sample.intros) + '\n') if len(self.sample.intros) > 0 else '\n') + \
            f'(answer : {self.sample.formal_answer_type}) ' + \
            (('\n' + format_variable_sequence(self.sample.outros[:]) + '\n') if len(self.sample.outros) > 0 else '\n') + \
            f': False := sorry'
        
        units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER)+'\n'+self.formal_problem_framework)
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), f'formal_problem_framework:{(self.sample.header or OPEN_HEADER)}\n{self.formal_problem_framework}'
        init_solution_state = units[-1].goal_state
        assert len(init_solution_state.goals) == 1 and [g.name for g in init_solution_state.goals] == [None], f'Invalid solving_state:\n{init_solution_state}'
        
        return init_solution_state
        
    init_forward_reasoning_state = to_sync(init_forward_reasoning_state_async)

    async def prove_eq_async(self, submission: str) -> Optional[str]:
        await self.server.restart_async()
        rpe_code = 'example' + \
            ((' ' + format_variable_sequence(self.sample.intros) + '\n') if len(self.sample.intros) > 0 else '\n') + \
            f' (answer : {self.sample.formal_answer_type}) : (\n{submission}\n) ↔ (\n{self.sample.formal_answer}\n)' + ' := sorry\n'
        
        units = await self.server.load_sorry_async((self.sample.header or OPEN_HEADER)+'\n'+rpe_code)
        assert len(units) >= 1 and 'error' not in str([x.messages for x in units]), f'formal_problem_framework:\n{(self.sample.header or OPEN_HEADER)}\n{rpe_code}\nmessage:\n{str([x.messages for x in units])}'
        init_state = units[-1].goal_state
        
        for tac in RPE_TACTICS:
            try:
                state = await self.server.goal_tactic_async(init_state, 0, tac)
                if state.is_solved:
                    return tac
            except TacticFailure:
                pass
        return None

    prove_eq = to_sync(prove_eq_async)
