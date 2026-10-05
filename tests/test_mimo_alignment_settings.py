"""Verify alignment-only component selection and configuration without Qt apps."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).parents[1]


def extract_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class MiMoAlignmentSettingsTests(unittest.TestCase):
    def test_manager_alignment_only_excludes_recognition_and_vad(self):
        path = "videocaptioner/ui/components/Qwen3ASRSettingWidget.py"
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        manager = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3ASRManagerDialog")
        init = next(n for n in manager.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        guard = next(n for n in init.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "aligner_only")
        namespace = {"self": SimpleNamespace(components=tuple({"kind": k} for k in ("runtime", "asr", "aligner", "firered-vad")))}
        exec(compile(ast.Module(body=guard.body, type_ignores=[]), path, "exec"), namespace)
        self.assertEqual([c["kind"] for c in namespace["self"].components], ["runtime", "aligner"])

    def test_mimo_constructor_receives_independent_alignment_fields(self):
        from videocaptioner.core.entities import TranscribeConfig
        config = TranscribeConfig(mimo_aligner_device="cpu", mimo_aligner_model_dir="aligner-only", mimo_aligner_runtime_python="isolated-python")
        factory = Mock()
        create = extract_function("videocaptioner/core/asr/transcribe.py", "_create_mimo_asr", {"TranscribeConfig": TranscribeConfig, "MiMoASR": factory})
        create("audio.wav", config)
        kwargs = factory.call_args.kwargs
        self.assertEqual(kwargs["aligner_device"], "cpu")
        self.assertEqual(kwargs["aligner_model_dir"], "aligner-only")
        self.assertEqual(kwargs["aligner_runtime_python"], "isolated-python")
        self.assertNotIn("qwen_asr_model_dir", kwargs)

    def test_both_settings_surfaces_use_shared_alignment_widget(self):
        for path in ("videocaptioner/ui/components/MiMoASRSettingWidget.py", "videocaptioner/ui/view/setting_interface.py"):
            tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "MiMoAlignmentSettingWidget"]
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
