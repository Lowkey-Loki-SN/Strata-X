"""moex (DSV41_BACKLOG #57): the learned expert profile is used in place of the config's only on the pack it was learned on.

The family engine writes "; pack fnv64:<FNV-1a 64 of the pack's experts.tsv>" into a profile's first line; learned_profile_args
compares it with the config's --pack, and starts from the learned profile also when the config's own file is missing."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serve.server import learned_profile_args, missing_profile_dropped, pack_digest, profile_pack, profile_shape  # noqa: E402

HEAD = "# strata-x expert profile: n_layer 40 n_expert 384, 15360 experts, 10 decode routings this run, 5 in VRAM (layer expert count)"


def profile(path: Path, pack: str | None = None) -> None:
    path.write_text(HEAD + (f"; pack {pack}" if pack else "") + "\n14 146 1.0\n", encoding="utf-8")


class ProfilePack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        (self.d / "pack").mkdir()
        (self.d / "pack" / "experts.tsv").write_bytes(b"a")
        self.base, self.learned = self.d / "base.txt", self.d / "learned.txt"
        profile(self.base)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self):
        cfg = {"expert_profile_save": str(self.learned), "cwd": str(self.d)}
        return learned_profile_args(cfg, ["--pack", str(self.d / "pack"), "--expert-profile", str(self.base)])

    def used(self, args):
        return args[args.index("--expert-profile") + 1]

    def test_digest_is_fnv1a_64(self):   # the engine's: FNV-1a 64 of the bytes ("a" -> af63dc4c8601ec8c)
        self.assertEqual(pack_digest(["--pack", str(self.d / "pack")], "."), "fnv64:af63dc4c8601ec8c")

    def test_same_pack(self):
        profile(self.learned, "fnv64:af63dc4c8601ec8c")
        self.assertEqual(profile_pack(str(self.learned)), "fnv64:af63dc4c8601ec8c")
        self.assertEqual(self.used(self.args()), str(self.learned))

    def test_digest_of_the_older_builds(self):   # their offset basis lost a digit: "a" -> 44bd8ad473cd9906, the same pack
        self.assertEqual(pack_digest(["--pack", str(self.d / "pack")], ".", 1469598103934665603), "fnv64:44bd8ad473cd9906")
        profile(self.learned, "fnv64:44bd8ad473cd9906")
        self.assertEqual(self.used(self.args()), str(self.learned))

    def test_another_pack(self):
        profile(self.learned, "fnv64:0000000000000001")
        self.assertEqual(self.used(self.args()), str(self.base))

    def test_older_profile_without_a_pack(self):   # as before: the shapes decide
        profile(self.learned)
        self.assertEqual(self.used(self.args()), str(self.learned))

    def test_config_profile_missing(self):
        profile(self.learned, "fnv64:af63dc4c8601ec8c")
        self.base.unlink()
        self.assertEqual(self.used(self.args()), str(self.learned))

    def test_no_learned_profile(self):
        self.assertEqual(self.used(self.args()), str(self.base))

    def test_first_start_from_a_template(self):   # neither the config's profile nor a learned one there yet
        self.base.unlink()
        args = missing_profile_dropped({"cwd": str(self.d), "expert_profile_save": str(self.learned)}, self.args())
        self.assertNotIn("--expert-profile", args)
        self.assertEqual(args[args.index("--expert-profile-save") + 1], str(self.learned))   # the run still saves one
        self.assertIn("--pack", args)

    def test_a_profile_that_is_there_is_kept(self):
        args = self.args()
        self.assertEqual(missing_profile_dropped({"cwd": str(self.d)}, args), args)
        relative = ["--expert-profile", "base.txt"]                     # relative to the config's cwd
        self.assertEqual(missing_profile_dropped({"cwd": str(self.d)}, relative), relative)
        self.assertEqual(missing_profile_dropped({"cwd": str(self.d)}, ["--expert-profile", "none.txt"]), [])

    def test_a_profile_from_before_the_rename(self):   # strata-glm's header (the engine's name until 2026-11-26)
        self.learned.write_text(HEAD.replace("strata-x", "strata-glm") + "\n14 146 1.0\n", encoding="utf-8")
        self.assertEqual(profile_shape(str(self.learned)), (40, 384))
        self.assertEqual(profile_shape(str(self.base)), (40, 384))


if __name__ == "__main__":
    unittest.main(verbosity=1)
