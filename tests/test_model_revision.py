# SPDX-License-Identifier: AGPL-3.0-or-later

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from heretic.config import Settings
from heretic.model import Model, get_model_class
from heretic.utils import create_reproduce_folder, resolve_model_revision

COMMIT = "a" * 40


class ModelRevisionTests(unittest.TestCase):
    def test_resolves_moving_reference_once(self) -> None:
        with patch("heretic.utils.huggingface_hub.model_info") as info:
            info.return_value = SimpleNamespace(sha=COMMIT)
            self.assertEqual(resolve_model_revision("owner/model", "main"), COMMIT)
            info.assert_called_once_with("owner/model", revision="main")

    def test_explicit_commit_needs_no_network(self) -> None:
        with patch("heretic.utils.huggingface_hub.model_info") as info:
            self.assertEqual(resolve_model_revision("owner/model", COMMIT), COMMIT)
            info.assert_not_called()

    def test_local_model_needs_no_hub_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("heretic.utils.huggingface_hub.model_info") as info:
                self.assertIsNone(resolve_model_revision(directory, None))
                info.assert_not_called()

    def test_missing_or_mutable_commit_is_rejected(self) -> None:
        for commit in (None, "main"):
            with self.subTest(commit=commit):
                with patch("heretic.utils.huggingface_hub.model_info") as info:
                    info.return_value = SimpleNamespace(sha=commit)
                    with self.assertRaisesRegex(ValueError, "immutable revision"):
                        resolve_model_revision("owner/model", None)

    def test_pin_is_used_for_first_tokenizer_load(self) -> None:
        settings = Settings.model_construct(model="owner/model", model_commit=None)
        with (
            patch("heretic.model.resolve_model_revision", return_value=COMMIT),
            patch("heretic.model.AutoTokenizer.from_pretrained") as load_tokenizer,
        ):
            # Stop before loading any model tensors or making further requests.
            load_tokenizer.side_effect = RuntimeError("stop after tokenizer")
            with self.assertRaisesRegex(RuntimeError, "stop after tokenizer"):
                Model(settings)
            self.assertEqual(settings.model_commit, COMMIT)
            load_tokenizer.assert_called_once_with("owner/model", revision=COMMIT)

    def test_model_class_uses_the_same_revision(self) -> None:
        with patch("heretic.model.PretrainedConfig.get_config_dict") as config:
            config.return_value = ({}, {})
            get_model_class("owner/model", COMMIT)
            config.assert_called_once_with("owner/model", revision=COMMIT)

    def test_switching_models_uses_the_new_revision(self) -> None:
        model = Model.__new__(Model)
        model.settings = Settings.model_construct(
            model="owner/base", model_commit=COMMIT
        )
        self.assertEqual(model.revision_kwargs, {"revision": COMMIT})
        model.settings.model = "owner/evaluation"
        model.settings.model_commit = "b" * 40
        self.assertEqual(model.revision_kwargs, {"revision": "b" * 40})
        model.settings.model_commit = None
        self.assertEqual(model.revision_kwargs, {})

    def test_reproduction_preserves_the_loaded_commit(self) -> None:
        settings = Settings.model_construct(model="owner/model", model_commit=COMMIT)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("heretic.utils.huggingface_hub.model_info") as info,
            patch("heretic.utils.generate_requirements_txt", return_value=""),
            patch("heretic.utils.generate_config_toml", return_value=""),
            patch(
                "heretic.utils.generate_reproduce_json", return_value="{}"
            ) as manifest,
            patch("heretic.utils.generate_reproduce_readme", return_value=""),
        ):
            create_reproduce_folder(
                Path(directory),
                settings,
                [],
                "missing-checkpoint",
                MagicMock(),
                {},
                False,
            )
            self.assertEqual(settings.model_commit, COMMIT)
            self.assertEqual(manifest.call_args.args[0].model_commit, COMMIT)
            info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
