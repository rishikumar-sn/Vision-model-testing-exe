import tempfile
import unittest
from pathlib import Path

from adapters import adapter_classes_for
from adapters.base import AdapterError
from adapters.onnx_adapter import ONNXAdapter
from adapters.openvino_adapter import OpenVINOAdapter
from adapters.ultralytics_adapter import UltralyticsAdapter


class OpenVINOSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(suffix="_openvino_model")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.xml = self.directory / "beadfinder.xml"
        for name in ("beadfinder.xml", "beadfinder.bin", "metadata.yaml"):
            (self.directory / name).touch()

    def test_folder_and_xml_choose_openvino(self):
        for source in (self.directory, self.xml):
            self.assertEqual(adapter_classes_for(str(source), "Auto"), [OpenVINOAdapter])
            self.assertEqual(OpenVINOAdapter.model_file(str(source)), self.xml)

    def test_missing_weights_explains_required_companion_file(self):
        self.xml.with_suffix(".bin").unlink()
        with self.assertRaisesRegex(AdapterError, "weights are missing"):
            OpenVINOAdapter.model_file(str(self.xml))

    def test_missing_metadata_explains_export_requirement(self):
        (self.directory / "metadata.yaml").unlink()
        with self.assertRaisesRegex(AdapterError, "metadata.yaml"):
            OpenVINOAdapter.model_file(str(self.xml))

    def test_multiple_models_are_not_silently_misidentified(self):
        (self.directory / "other.xml").touch()
        for source in (self.directory, self.xml):
            with self.assertRaisesRegex(AdapterError, "exactly one"):
                OpenVINOAdapter.model_file(str(source))

    def test_existing_onnx_and_torch_routing(self):
        self.assertEqual(adapter_classes_for("model.onnx", "Auto"), [UltralyticsAdapter, ONNXAdapter])
        self.assertIs(adapter_classes_for("model.pt", "Auto")[0], UltralyticsAdapter)


if __name__ == "__main__":
    unittest.main()
