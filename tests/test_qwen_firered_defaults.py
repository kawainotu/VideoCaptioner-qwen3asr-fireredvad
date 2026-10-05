"""Qwen FireRed defaults and restore tests without loading personal settings."""

import ast
import inspect
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from qfluentwidgets import QConfig

from videocaptioner.core.asr.qwen3_asr import Qwen3ASR
from videocaptioner.core.asr.qwen3_asr_runner import build_parser
from videocaptioner.core.entities import TranscribeConfig
from videocaptioner.core.qwen3_vad_defaults import FIRERED_VAD_DEFAULTS
from videocaptioner.ui.common.qwen_vad_defaults import reset_qwen_firered_defaults

ROOT = Path(__file__).parents[1]
EXPECTED = dict(smooth_window_size=5, speech_threshold=.4, min_speech_frame=20,
                max_speech_frame=2000, min_silence_frame=20, merge_silence_frame=0,
                extend_speech_frame=0, chunk_max_frame=30000)


def fresh_config():
    """Compile definitions only; omit global cfg creation and settings loading."""
    source = ROOT / "videocaptioner/ui/common/config.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            break
    namespace = {"__name__": "isolated_qwen_config"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["Config"]()


class QwenFireRedDefaultsTests(unittest.TestCase):
    def test_official_defaults_match_all_layers(self):
        self.assertEqual(asdict(FIRERED_VAD_DEFAULTS), EXPECTED)
        config = fresh_config()
        entity = TranscribeConfig()
        signature = inspect.signature(Qwen3ASR)
        args = build_parser().parse_args(["--audio", "a.wav", "--output", "o.json",
                                         "--asr-model", "asr", "--aligner-model", "aligner"])
        for name, value in EXPECTED.items():
            self.assertEqual(getattr(config, "qwen_asr_firered_vad_" + name).value, value)
            self.assertEqual(getattr(entity, "qwen_asr_firered_vad_" + name), value)
            self.assertEqual(signature.parameters["firered_vad_" + name].default, value)
            self.assertEqual(getattr(args, "firered_vad_" + name), value)

    def test_load_preserves_saved_custom_and_previous_tuned_values(self):
        for saved in ({"FireRedVadMinSpeechFrame": 12, "FireRedVadMaxSpeechFrame": 1500,
                       "FireRedVadMinSilenceFrame": 30, "FireRedVadMergeSilenceFrame": 50,
                       "FireRedVadExtendSpeechFrame": 15}, {"FireRedVadMinSpeechFrame": 33}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "settings.json"
                path.write_text(json.dumps({"QwenASR": saved}), encoding="utf-8")
                original_bytes = path.read_bytes()
                config = fresh_config()
                QConfig().load(str(path), config)
                self.assertEqual(path.read_bytes(), original_bytes)
                for name in EXPECTED:
                    item = getattr(config, "qwen_asr_firered_vad_" + name)
                    self.assertEqual(item.value, saved.get(item.name, EXPECTED[name]))

    def test_restore_sets_only_eight_qwen_parameters(self):
        config = SimpleNamespace(set=Mock())
        for name in EXPECTED:
            setattr(config, "qwen_asr_firered_vad_" + name, "qwen:" + name)
        config.mimo_vad_filter = object()
        config.qwen_asr_vad_model = object()
        config.qwen_asr_vad_threshold = object()
        reset_qwen_firered_defaults(config)
        self.assertEqual(config.set.call_args_list,
                         [unittest.mock.call("qwen:" + name, value) for name, value in EXPECTED.items()])

    def test_config_startup_only_loads_without_migration(self):
        tree = ast.parse((ROOT / "videocaptioner/ui/common/config.py").read_text(encoding="utf-8"))
        config_class_index = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.ClassDef) and n.name == "Config")
        tail = tree.body[config_class_index + 1:]
        calls = [n for root in tail for n in ast.walk(root) if isinstance(n, ast.Call)]
        self.assertEqual([ast.unparse(call.func) for call in calls], ["Config", "QColor", "qconfig.load"])


if __name__ == "__main__":
    unittest.main()
