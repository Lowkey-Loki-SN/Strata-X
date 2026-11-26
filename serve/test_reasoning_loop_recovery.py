"""CPU-only regression for opt-in reasoning recovery: python -m unittest serve.test_reasoning_loop_recovery."""
import contextlib
import dataclasses
import io
import json
import math
import os
import random
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from serve.server import (Service, ByteTokenizer, MockEngine, HIGH_EFFORT,
                          reasoning_repeat_coverage, focused_recovery_prompt, held_repeats, periodic_tail)


class Tokenizer(ByteTokenizer):
    def token_bytes(self, token):
        return self.SPECIALS[token - 256].encode() if token >= 256 else bytes([token])


class RecordingEngine(MockEngine):
    def __init__(self, tokenizer, scripts):
        super().__init__(tokenizer, scripts, max_context=100000)
        self.calls = []
        self.sampling_calls = []
        self.sampling_objects = []

    def generate(self, ids, max_new, sampling, cancel, **kwargs):
        self.calls.append((list(ids), max_new))
        self.sampling_calls.append(dict(sampling or {}))
        self.sampling_objects.append(sampling)
        yield from super().generate(ids, max_new, sampling, cancel, **kwargs)


class ReasoningLoopRecovery(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        self.ids = self.tok.encode('<|im_start|>system\n' + HIGH_EFFORT +
            '<|im_end|>\n<|im_start|>user\nSolve this unchanged task<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n', parse_special=True)
        self.loop = 'Need maybe Agent support agent seven b. ' * 700

    def test_repetition_and_novel_work(self):
        self.assertEqual(reasoning_repeat_coverage(' '.join(str(i) for i in range(4000))), 0)
        self.assertGreater(reasoning_repeat_coverage(self.loop), .9)

    def test_a_loop_written_as_one_long_word(self):
        """#1753: a 24k-digit string cycling an 11-digit pattern is one word to the split and never one token twice."""
        self.assertEqual(reasoning_repeat_coverage('Counting the digits of 9^104. ' + '58747693387' * 3000), 1)
        counted = 'Digits: ' + ''.join(str(i) for i in range(1000))     # 2,893 chars that never repeat
        self.assertEqual(reasoning_repeat_coverage(counted), 0)
        self.assertEqual(reasoning_repeat_coverage('58747693387' * 100), 0)   # a tail too short to judge

    def test_user_instruction_is_never_rewritten(self):
        ids = self.tok.encode('<|im_start|>user\n' + HIGH_EFFORT, parse_special=True)
        self.assertIsNone(focused_recovery_prompt(self.tok, ids, []))

    def test_preserves_task_and_exact_generated_tokens(self):
        generated = self.tok.encode('中文 partially checked code', parse_special=False)
        prompt = focused_recovery_prompt(self.tok, self.ids, generated)
        self.assertEqual(prompt[-len(generated):], generated)
        self.assertIn('Solve this unchanged task', self.tok.decode(prompt))
        self.assertNotIn('</think>', self.tok.decode(prompt))

    def run_case(self, enabled):
        engine = RecordingEngine(self.tok, [self.loop,
            'I have verified the solution.\n</think>\nComplete solution.'])
        service = Service(engine, self.tok, None)
        service.reasoning_loop_recovery = enabled
        events = list(service.run(self.ids, True, [], 40000, {}, threading.Event()))
        return engine, events

    def test_old_explicit_sampler_recovers_without_mutating_request(self):
        engine = RecordingEngine(self.tok, [self.loop,
            'Verified.\n</think>\nComplete solution.'])
        service = Service(engine, self.tok, None)
        service.reasoning_loop_recovery = True
        request = {'temperature': .2, 'presence_penalty': 0, 'top_p': .95, 'top_k': 64, 'seed': 2}
        list(service.run(self.ids, True, [], 40000, request, threading.Event()))
        self.assertEqual(engine.sampling_calls[0], request)
        self.assertEqual(engine.sampling_calls[1]['temperature'], 1.0)
        self.assertEqual(engine.sampling_calls[1]['presence_penalty'], 1.5)
        self.assertEqual(engine.sampling_calls[1]['top_k'], 64)      # a client's top_k and top_p are never touched
        self.assertEqual(engine.sampling_calls[1]['top_p'], .95)
        self.assertEqual(engine.sampling_calls[1]['seed'], request['seed'])
        self.assertEqual(request['temperature'], .2)
        self.assertEqual(request['presence_penalty'], 0)

    def test_frozen_cohort_sampler_is_unchanged_during_recovery(self):
        engine = RecordingEngine(self.tok, [self.loop, 'Done.\n</think>\nComplete solution.'])
        service = Service(engine, self.tok, None)
        service.reasoning_loop_recovery = True
        sampler = {'temperature': 1.0, 'top_p': .95, 'top_k': 20, 'presence_penalty': 1.5, 'seed': 2}
        list(service.run(self.ids, True, [], 40000, sampler, threading.Event()))
        self.assertEqual(len(engine.sampling_objects), 2)
        self.assertIs(engine.sampling_objects[0], engine.sampling_objects[1])
        self.assertIs(engine.sampling_objects[1], sampler)

    def test_a_lower_top_p_and_top_k_are_never_raised_or_lowered(self):
        engine = RecordingEngine(self.tok, [self.loop, 'Verified.\n</think>\nComplete solution.'])
        service = Service(engine, self.tok, None)
        service.reasoning_loop_recovery = "recover"
        list(service.run(self.ids, True, [], 40000, {'temperature': 0, 'top_p': .5, 'top_k': 5}, threading.Event()))
        self.assertEqual((engine.sampling_calls[1]['top_p'], engine.sampling_calls[1]['top_k']), (.5, 5))
        self.assertEqual((engine.sampling_calls[1]['temperature'], engine.sampling_calls[1]['presence_penalty']), (1.0, 1.5))

    def test_the_splice_keeps_a_literal_think_tag_as_text(self):
        """#537: the prompt is spliced as token ids, never decoded and encoded again."""
        from serve.frontend import THINK_TAGS
        head = '<|im_start|>system\n' + HIGH_EFFORT
        marked = self.tok.encode(head + '<|im_end|>\n<|im_start|>user\nquote ' + THINK_TAGS['</think>'] +
                                 ' please<|im_end|>\n<|im_start|>assistant\n<think>\n', parse_special=True)
        spliced = focused_recovery_prompt(self.tok, marked, [])
        tail = marked[len(self.tok.encode(head, parse_special=True)):]
        self.assertEqual(spliced[-len(tail):], tail)             # everything after the sentence is the same ids
        self.assertNotEqual(spliced, marked)

    def test_stop_mode_ends_the_reply_at_the_loop(self):
        engine = RecordingEngine(self.tok, [self.loop])
        service = Service(engine, self.tok, None)
        service.reasoning_loop_recovery = "stop"
        events = list(service.run(self.ids, True, [], 40000, {}, threading.Event()))
        self.assertEqual(len(engine.calls), 1)                   # no second pass
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertLess(events[-1][1]['completion_tokens'], 20000)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 0)

    def test_stop_mode_ends_a_loop_written_as_one_long_word(self):
        """#1753: the word count never sees the digits, so only the tail's period ends the reply."""
        engine = RecordingEngine(self.tok, ['9^104 = ' + '58747693387' * 3000])
        service = Service(engine, self.tok, None)
        service.reasoning_loop_recovery = "stop"
        events = list(service.run(self.ids, True, [], 40000, {}, threading.Event()))
        self.assertEqual(len(engine.calls), 1)                   # no second pass
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertLess(events[-1][1]['completion_tokens'], 5000)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 0)

    def test_disabled_is_one_pass(self):
        engine, events = self.run_case(False)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 0)

    def test_default_single_token_guard_does_not_detect_passage_loop(self):
        engine = RecordingEngine(self.tok, [self.loop])
        service = Service(engine, self.tok, None)
        events = list(service.run(self.ids, True, [], 16384, {}, threading.Event()))
        self.assertEqual(service.repeat_stop_tokens, 256)
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertEqual(events[-1][1]['completion_tokens'], 16384)
        self.assertFalse(any(e.kind == 'content' for kind, e in events if kind == 'event'))

    def test_recovery_is_once_and_never_extends_output_limit(self):
        engine = RecordingEngine(self.tok, [self.loop, self.loop])
        service = Service(engine, self.tok, None)
        service.reasoning_loop_recovery = True
        events = list(service.run(self.ids, True, [], 16384, {}, threading.Event()))
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertEqual(events[-1][1]['completion_tokens'], 16384)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 1)

    def test_repeated_final_text_does_not_change_sampler_or_prompt(self):
        engine = RecordingEngine(self.tok, [self.loop])
        service = Service(engine, self.tok, None)
        service.reasoning_loop_recovery = True
        events = list(service.run(self.ids, False, [], 40000, {}, threading.Event()))
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 0)

    def test_missing_native_high_hint_skips_recovery(self):
        engine = RecordingEngine(self.tok, [self.loop])
        service = Service(engine, self.tok, None)
        service.reasoning_loop_recovery = True
        ids = self.tok.encode('<|im_start|>user\nTask<|im_end|>\n', parse_special=True)
        events = list(service.run(ids, True, [], 40000, {}, threading.Event()))
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 0)

    def test_cancel_during_reasoning_does_not_start_recovery(self):
        engine = RecordingEngine(self.tok, [self.loop, 'Unused'])
        service = Service(engine, self.tok, None)
        service.reasoning_loop_recovery = True
        cancel = threading.Event()
        events = []
        for event in service.run(self.ids, True, [], 40000, {}, cancel):
            events.append(event)
            if event[0] == 'event':
                cancel.set()
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(events[-1][1]['finish'], 'cancel')
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 0)

    def test_batch_status_and_history_record_recovery(self):
        engine = RecordingEngine(self.tok, [self.loop, 'Done.\n</think>\nAnswer.'])
        engine.batch = 2
        service = Service(engine, self.tok, None)
        service.reasoning_loop_recovery = True
        events = list(service.run(self.ids, True, [], 40000, {}, threading.Event()))
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 1)
        self.assertEqual(service.history[-1]['reasoning_recoveries'], 1)
        self.assertGreaterEqual(service.history[-1]['reasoning_repeat_coverage'], .25)
        self.assertFalse(service.status['busy'])
        self.assertEqual(service.live_reqs, {})

    def test_natural_answer_with_shared_budget_and_preserved_prefix(self):
        engine, events = self.run_case(True)
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 1)
        self.assertEqual(events[-1][1]['finish'], 'stop')
        content = ''.join(e.text or '' for kind, e in events if kind == 'event' and e.kind == 'content')
        self.assertEqual(content, 'Complete solution.')
        head_size = len(focused_recovery_prompt(self.tok, self.ids, []))
        retained = engine.calls[1][0][head_size:]
        self.assertEqual(retained, self.tok.encode(self.loop)[:len(retained)])
        self.assertEqual(engine.calls[1][1], 40000 - len(retained))
        self.assertNotIn('</think>', self.tok.decode(engine.calls[1][0]))


LOOP = 'Need maybe Agent support agent seven b. ' * 700
# a long answer that repeats its structure on purpose: a 300-row table of different numbers, and repetitive code
TABLE = '| n | square | cube | sqrt |\n|---|---|---|---|\n' + ''.join(
    f'| {i} | {i * i} | {i ** 3} | {math.sqrt(i):.4f} |\n' for i in range(300))
TESTS = ''.join(f'    def test_case_{i}(self):\n        self.assertEqual(square({i}), {i * i})\n\n' for i in range(300))
SWITCH = 'switch (op) {\n' + ''.join(f'    case {i}:\n        result = handle_{i}(state);\n        break;\n'
                                     for i in range(300)) + '}\n'


class AnswerLoopRecovery(unittest.TestCase):
    """DSV41_BACKLOG #46: opt-in "answer_loop_recovery": "stop" - the thinking's coverage look on the answer, at half of the
    last 2,000 words (the thinking's: a quarter) or the periodic tail."""

    def setUp(self):
        self.tok = Tokenizer()
        self.ids = self.tok.encode('<|im_start|>user\nTask<|im_end|>\n<|im_start|>assistant\n<think>\n', parse_special=True)

    def run_answer(self, answer, mode='stop', thinking=True, max_new=60000):
        engine = RecordingEngine(self.tok, ['Short thought.\n</think>\n\n' + answer if thinking else answer])
        service = Service(engine, self.tok, None)
        service.answer_loop_recovery = mode
        with contextlib.redirect_stdout(io.StringIO()) as log:
            events = list(service.run(self.ids, thinking, [], max_new, {}, threading.Event()))
        content = ''.join(e.text or '' for kind, e in events if kind == 'event' and e.kind == 'content')
        return events[-1][1], content, log.getvalue()

    def test_an_answer_that_loops_is_stopped(self):
        for thinking in (True, False):
            with self.subTest(thinking=thinking):
                done, content, log = self.run_answer(LOOP, thinking=thinking)
                self.assertEqual(done['finish'], 'length')
                self.assertLess(done['completion_tokens'], 6000)      # of 28,000
                self.assertTrue(content and LOOP.startswith(content))
                self.assertIn('the answer repeated the same passages', log)
                self.assertIn('answer_loop_recovery', log)

    def test_a_loop_written_as_one_long_word(self):
        """#1753's tail check: one long word cycling an 11-digit pattern."""
        done, content, log = self.run_answer('9^104 = ' + '58747693387' * 3000)
        self.assertEqual(done['finish'], 'length')
        self.assertLess(done['completion_tokens'], 5000)

    def test_long_answers_with_repeated_structure_are_not_stopped(self):
        for name, answer in (('table', TABLE), ('tests', TESTS), ('switch', SWITCH)):
            with self.subTest(answer=name):
                self.assertGreater(len(re.findall(r'\w+|[^\w\s]', answer)), 2000)   # long enough to be measured
                self.assertLess(reasoning_repeat_coverage(answer), .5)
                done, content, log = self.run_answer(answer)
                self.assertEqual((done['finish'], content), ('stop', answer))
                self.assertNotIn('repeated', log)

    def test_off_is_the_whole_answer_as_before(self):
        for mode in (False, None):
            with self.subTest(mode=mode):
                done, content, log = self.run_answer(LOOP, mode=mode)
                self.assertEqual((done['finish'], content), ('stop', LOOP))
                self.assertNotIn('repeated', log)
        # the thinking's own guard does not look at the answer either
        engine = RecordingEngine(self.tok, ['Short thought.\n</think>\n\n' + LOOP])
        service = Service(engine, self.tok, None)
        service.reasoning_loop_recovery = 'stop'
        events = list(service.run(self.ids, True, [], 60000, {}, threading.Event()))
        self.assertEqual(events[-1][1]['finish'], 'stop')

    def test_the_thinking_is_not_the_answer(self):
        # a thinking that loops is reasoning_loop_recovery's to stop, not this key's
        engine = RecordingEngine(self.tok, [LOOP + '\n</think>\n\nThe answer.'])
        service = Service(engine, self.tok, None)
        service.answer_loop_recovery = 'stop'
        events = list(service.run(self.ids, True, [], 60000, {}, threading.Event()))
        self.assertEqual(events[-1][1]['finish'], 'stop')
        self.assertEqual(''.join(e.text for kind, e in events if kind == 'event' and e.kind == 'content'), 'The answer.')


D = '｜DSML｜'


def block(city):
    return (f'\n\n<{D} calls>\n<{D} invoke name="get_weather">\n<{D} parameter name="city" string="true">{city}</{D} parameter>\n'
            f'</{D} invoke>\n</{D} calls>')


BLOCK = block('Rome')
WEATHER = [{'name': 'get_weather', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}}}]


class HeldCallsInTheThinking(unittest.TestCase):
    """DSV41_BACKLOG #46 (b): a call block written inside the thinking waits in the parser (pending) until what follows it says
    whether it was an act; a thinking that repeats such blocks gave the reasoning look no text, so it never ran.  For
    DeepSeek-V4.1 (FamilySpec.loop_guard_held) the held blocks that repeat one held before them count (held_repeats);
    distinct calls never do."""

    def setUp(self):
        from serve import families
        self.spec = families.get('deepseek41')
        self.tok = Tokenizer()

    def run_calls(self, spec, script, mode='stop', answer_mode=False, coverage=None):
        engine = RecordingEngine(self.tok, [script])
        service = Service(engine, self.tok, None, family=spec)
        service.stop_ids = set(self.tok.encode('<|im_end|>', parse_special=True))   # the stub's end of a reply
        service.reasoning_loop_recovery = mode
        if coverage is not None:
            service.reasoning_loop_coverage = coverage
        service.answer_loop_recovery = answer_mode
        with contextlib.redirect_stdout(io.StringIO()) as log:
            events = list(service.run(self.tok.encode('prompt'), True, WEATHER, 200000, {}, threading.Event()))
        calls = [e for kind, e in events if kind == 'event' and e.kind == 'tool_call']
        return events[-1][1], calls, log.getvalue()

    def test_repeated_held_calls_are_stopped(self):
        script = 'Let me check the weather.' + BLOCK * 200 + '</think>Done.'
        done, calls, log = self.run_calls(self.spec, script)
        self.assertEqual(done['finish'], 'length')
        self.assertLess(done['completion_tokens'], len(self.tok.encode(script)) // 3)
        self.assertEqual(calls, [])                                   # cut inside the thinking: no call delivered
        self.assertIn('the thinking repeated the same passages', log)

    def test_repeated_held_calls_are_stopped_at_a_full_window(self):
        # moex (DSV41_PLAN §60): DeepSeek-V4.1's configs set reasoning_loop_coverage 1.0, and one block repeated still ends it
        script = 'Let me check the weather.' + BLOCK * 200 + '</think>Done.'
        done, calls, log = self.run_calls(self.spec, script, coverage=1.0)
        self.assertEqual(done['finish'], 'length')
        self.assertLess(done['completion_tokens'], len(self.tok.encode(script)) // 3)
        self.assertEqual(calls, [])
        self.assertIn('the thinking repeated the same passages', log)

    def test_distinct_held_calls_are_not_stopped(self):
        # 60 different calls held in one thinking: their DSML markup alone covers a window of them (counted whole, they read
        # as a loop and the thinking was cut), but no block repeats another, so none counts - they are delivered at </think>
        blocks = [block(f'City {i}') for i in range(60)]
        self.assertGreater(reasoning_repeat_coverage(''.join(blocks)), .9)
        self.assertEqual(held_repeats([[b, None] for b in blocks]), '')
        done, calls, log = self.run_calls(self.spec, 'Let me check the weather.' + ''.join(blocks) + '</think>Done.')
        self.assertEqual(done['finish'], 'stop')
        self.assertEqual([c.call.arguments['city'] for c in calls], [f'City {i}' for i in range(60)])
        self.assertNotIn('repeated the same passages', log)

    def test_repeats_among_other_calls_are_stopped(self):
        # two calls written in turn, and distinct calls followed by one written again and again
        for name, script in (('alternating', 'Checking.' + (block('Rome') + block('Oslo')) * 100 + '</think>Done.'),
                             ('distinct, then one', 'Checking.' + ''.join(block(f'City {i}') for i in range(60)) +
                              block('City 7') * 200 + '</think>Done.')):
            with self.subTest(script=name):
                done, calls, log = self.run_calls(self.spec, script)
                self.assertEqual(done['finish'], 'length')
                self.assertEqual(calls, [])
                self.assertIn('the thinking repeated the same passages', log)

    def test_held_repeats(self):
        a, b = block('Rome'), block('Oslo')
        pending = [[a, None], [b, None], [a + '\n\n', None], [' ' + b.strip(), None], [block('Bonn'), None], [a, None]]
        self.assertEqual(held_repeats(pending), a + '\n\n' + ' ' + b.strip() + a)    # each block's later copies, as held
        self.assertEqual(held_repeats([]), '')

    def test_other_families_as_before(self):
        # without the family's flag the look waits for the parser to hold nothing, which these blocks never let it
        script = 'Let me check the weather.' + BLOCK * 200 + '</think>Done.'
        done, calls, log = self.run_calls(dataclasses.replace(self.spec, loop_guard_held=False), script)
        self.assertEqual(done['finish'], 'stop')
        self.assertEqual(len(calls), 200)
        self.assertNotIn('repeated the same passages', log)

    def test_a_few_held_calls_are_delivered(self):
        done, calls, log = self.run_calls(self.spec, 'Let me check the weather.' + BLOCK * 3 + '</think>')
        self.assertEqual((done['finish'], len(calls)), ('stop', 3))
        self.assertEqual(calls[0].call.arguments, {'city': 'Rome'})

    def test_the_answer_guard_with_this_family(self):
        # the family's parser gives the answer as written; a looping one is stopped there too
        done, calls, log = self.run_calls(self.spec, 'Short.</think>' + LOOP, mode=False, answer_mode='stop')
        self.assertEqual(done['finish'], 'length')
        self.assertIn('the answer repeated the same passages', log)
        done, calls, log = self.run_calls(self.spec, 'Short.</think>' + TABLE, mode=False, answer_mode='stop')
        self.assertEqual(done['finish'], 'stop')


class _Reached(Exception):
    """raised by the patched serve(): main() got past the config's checks."""


class AnswerLoopRecoveryConfig(unittest.TestCase):
    """The config key: false (the default) or "stop"; anything else stops the start with its name."""

    def run_main(self, cfg):
        import serve.server as S
        seen = {}

        def fake_serve(svc, host=None, port=None):
            seen['svc'] = svc
            raise _Reached

        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ), \
                mock.patch.object(S, 'serve', fake_serve), contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(sys, 'stderr', new=io.StringIO()):
            os.environ.pop('STRATA_API_KEY', None)
            p = Path(d) / 'cfg.json'
            p.write_text(json.dumps(cfg), encoding='utf-8')
            with mock.patch.object(sys, 'argv', ['server.py', '--engine', 'mock', '--port', '0', '--config', str(p)]):
                try:
                    S.main()
                except _Reached:
                    pass
        return seen['svc']

    def test_values(self):
        self.assertIs(self.run_main({}).answer_loop_recovery, False)
        self.assertIs(self.run_main({'answer_loop_recovery': False}).answer_loop_recovery, False)
        self.assertEqual(self.run_main({'answer_loop_recovery': 'stop'}).answer_loop_recovery, 'stop')
        for bad in (True, 'recover', 'yes', 1, None):
            with self.subTest(value=bad), self.assertRaises(SystemExit) as cm:
                self.run_main({'answer_loop_recovery': bad})
            self.assertIn('answer_loop_recovery', str(cm.exception))


def number_drafts(seed, n=12, rows=24):
    """Drafts of a table as rows of numbers, each changing four rows of the one before it, with words between them: work,
    not a loop."""
    r = random.Random(seed)
    values = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76]
    cur = [[r.choice([0.5, 1, 1.5]), r.choice(values)] for _ in range(rows)]
    out = []
    for k in range(n):
        for _ in range(4):
            cur[r.randrange(rows)] = [r.choice([0.5, 1, 1.5]), r.choice(values)]
        t, lines = 0.0, []
        for d, v in cur:
            lines.append(f"[{t:g}, {d:g}, {v}]")
            t += d
        weights = ", ".join(r.choice(["1.0", "1.0", "1.0", "0.98", "0.95"]) for _ in range(24))
        out.append(f"Draft {k + 1} of the table, with a change in row {k + 2}: " + ", ".join(lines) +
                   f". Weights: {weights}. Checking the sums and the order of draft {k + 1} against the totals.\n")
    return "".join(out)


class ReasoningLoopCoverage(unittest.TestCase):
    """moex (DSV41_PLAN §60): "reasoning_loop_coverage", the share of repeats that ends the thinking (0.25 by default).  The
    look counts every digit, comma, period and bracket as a word, so rows of numbers that share values read as repeats;
    at 1.0 only a window of nothing but repeats ends it - a loop of words or of numbers, or the periodic tail."""

    def setUp(self):
        self.tok = Tokenizer()
        self.ids = self.tok.encode('<|im_start|>user\nFill in the table<|im_end|>\n<|im_start|>assistant\n<think>\n',
                                   parse_special=True)

    def run_reasoning(self, text, coverage=None, max_new=200000):
        engine = RecordingEngine(self.tok, [text + '\n</think>\nThe table.'])
        service = Service(engine, self.tok, None)
        service.reasoning_loop_recovery = 'stop'
        if coverage is not None:
            service.reasoning_loop_coverage = coverage
        with contextlib.redirect_stdout(io.StringIO()) as log:
            events = list(service.run(self.ids, True, [], max_new, {}, threading.Event()))
        content = ''.join(e.text for kind, e in events if kind == 'event' and e.kind == 'content')
        return events[-1][1], content, log.getvalue()

    def test_number_drafts_read_as_repeats(self):
        for seed in range(3):
            with self.subTest(seed=seed):
                self.assertTrue(.5 < reasoning_repeat_coverage(number_drafts(seed)) < .8)

    def test_the_default_stops_them_and_a_full_window_does_not(self):
        done, content, log = self.run_reasoning(number_drafts(0))
        self.assertEqual(done['finish'], 'length')
        self.assertIn('the thinking repeated the same passages', log)
        done, content, log = self.run_reasoning(number_drafts(0), coverage=1.0)
        self.assertEqual((done['finish'], content), ('stop', 'The table.'))
        self.assertNotIn('repeated', log)

    def test_a_full_window_still_stops_loops(self):
        words = ' '.join(f'step{i}' for i in range(80)) + '. '          # 551 chars: too long a period for the tail
        numbers = ', '.join(f'[{i % 7}, {i * 3 % 11}]' for i in range(40)) + '.\n'
        for name, loop in (('words', words * 100), ('numbers', numbers * 100),
                           ('the tail', 'Need maybe Agent support agent seven b. ' * 700)):
            with self.subTest(loop=name):
                done, content, log = self.run_reasoning('Starting. ' + loop, coverage=1.0)
                self.assertEqual((done['finish'], content), ('length', ''))
                self.assertLess(done['completion_tokens'], len(loop) // 2)
                self.assertIn('reasoning_loop_coverage 1', log)

    def test_a_guard_stop_is_not_called_max_tokens(self):
        hint = 'reached max tokens while still thinking'
        done, content, log = self.run_reasoning('Need maybe Agent support agent seven b. ' * 700)
        self.assertEqual(done['finish'], 'length')
        self.assertNotIn(hint, log)
        done, content, log = self.run_reasoning(number_drafts(0), coverage=1.0, max_new=1000)   # max tokens itself
        self.assertEqual(done['finish'], 'length')
        self.assertIn(hint, log)


class ReasoningLoopCoverageConfig(unittest.TestCase):
    """The config key: a number above 0 and at most 1 (0.25 when absent); anything else stops the start with its name."""

    def test_values(self):
        run_main = AnswerLoopRecoveryConfig.run_main
        self.assertEqual(run_main(self, {}).reasoning_loop_coverage, .25)
        self.assertEqual(run_main(self, {'reasoning_loop_coverage': 1.0}).reasoning_loop_coverage, 1.0)
        self.assertEqual(run_main(self, {'reasoning_loop_coverage': 1}).reasoning_loop_coverage, 1.0)
        for bad in (0, -0.5, 1.5, True, '1.0', None):
            with self.subTest(value=bad), self.assertRaises(SystemExit) as cm:
                run_main(self, {'reasoning_loop_coverage': bad})
            self.assertIn('reasoning_loop_coverage', str(cm.exception))


if __name__ == '__main__':
    unittest.main()


class PeriodicTail(unittest.TestCase):
    """moex (DSV41_BACKLOG #64): the call guard's only look - a period of at most 64 chars over the last 2,048."""

    def test_a_loop_is_periodic(self):
        self.assertTrue(periodic_tail("x = 1\n" + "    self.assertEqual(a, b)\n" * 120))
        self.assertTrue(periodic_tail("0," * 2000))

    def test_repetitive_code_is_not(self):
        # tests repeating their setup (#46 measured 0.95 coverage on such text): each block differs, so no short period
        code = "".join(f"    def test_{i}(self):\n        x = make({i})\n        self.assertEqual(x.value, {i * 7})\n\n"
                       for i in range(60))
        self.assertFalse(periodic_tail(code))
        self.assertGreaterEqual(reasoning_repeat_coverage(code), 0.0)

    def test_short_text_is_not(self):
        self.assertFalse(periodic_tail("ab" * 100))

