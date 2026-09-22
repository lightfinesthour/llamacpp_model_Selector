import importlib.util
import struct
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('updater', Path(__file__).with_name('update-llamacpp.py'))
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


class UpdaterTests(unittest.TestCase):
    @patch.object(u.platform, 'machine', return_value='AMD64')
    def test_pairs_by_arch_and_cuda_version(self, _):
        names = ['llama-b10981-bin-win-cuda-13.4-arm64.zip',
                 'cudart-llama-bin-win-cuda-13.4-arm64.zip',
                 'llama-b10981-bin-win-cuda-14.0-x64.zip',
                 'llama-b10981-bin-win-cuda-13.4-x64.zip',
                 'cudart-llama-bin-win-cuda-13.4-x64.zip']
        pair = u.find_cuda_assets([{'name': n} for n in names])
        self.assertEqual([a['name'] for a in pair], names[-2:])
        self.assertEqual(u.find_cuda_assets([{'name': n} for n in names[:3]]), (None, None))

    @patch.object(u.platform, 'machine', return_value='AMD64')
    def test_rejects_arm_and_malformed_binary(self, _):
        data = bytearray(128)
        data[:2] = b'MZ'
        struct.pack_into('<I', data, 60, 64)
        data[64:68] = b'PE\0\0'
        struct.pack_into('<H', data, 68, 0x8664)
        u.validate_pe(data, 'valid.exe')
        struct.pack_into('<H', data, 68, 0xaa64)
        with self.assertRaises(RuntimeError):
            u.validate_pe(data, 'arm.exe')
        with self.assertRaises(RuntimeError):
            u.validate_pe(b'bad', 'bad.exe')


if __name__ == '__main__':
    unittest.main()
