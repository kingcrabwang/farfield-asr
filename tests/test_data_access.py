"""在内存 ZIP 上验证字节读取与 seek，测试不依赖外网或任何访问凭证。"""

import io
import zipfile
from types import SimpleNamespace

from farfield_asr.data_access import RangeReader, SharedRangeCache


def test_remote_zip_interface_reads_exact_members():
    buffer = io.BytesIO()
    payload = bytes(range(256)) * 100
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("test.wav", payload)
        archive.writestr("transcript.txt", "你好")
    archive_bytes = buffer.getvalue()

    class Bucket:
        def head_object(self, key):
            return SimpleNamespace(content_length=len(archive_bytes))

        def get_object(self, key, byte_range):
            start, end = byte_range
            return io.BytesIO(archive_bytes[start:end + 1])

    cache = SharedRangeCache(Bucket(), "test", block_size=97, max_blocks=4)
    with zipfile.ZipFile(RangeReader(cache)) as archive:
        assert archive.read("test.wav") == payload
        assert archive.read("transcript.txt").decode() == "你好"
