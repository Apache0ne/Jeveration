"""Regression: geometry-only workflows must not eagerly load a full VLM."""
import subprocess
import sys
import unittest


class TestLazyPackageImports(unittest.TestCase):
    def test_simple_package_import_does_not_import_pipeline(self):
        code = (
            "import sys, jeveration; "
            "assert 'jeveration.pipeline' not in sys.modules; "
            "assert 'jeveration.backbone' not in sys.modules; "
            "assert 'JeverationPipeline' in jeveration.__all__; "
            "print('lazy package import verified')"
        )
        result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,msg=result.stdout+result.stderr)

    def test_bootstrap_module_does_not_eagerly_load_torch(self):
        code=(
            "import sys, jeveration.rgbd_gt_launch; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "print('lightweight bootstrap verified')"
        )
        result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,msg=result.stdout+result.stderr)


if __name__=="__main__":
    unittest.main()
