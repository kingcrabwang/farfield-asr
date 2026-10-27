"""公开 OSS ZIP 的按需字节读取；支持 zipfile，不下载无关音频。"""

import io
import subprocess
import threading
from collections import OrderedDict
from types import SimpleNamespace


class HTTPRangeBucket:
    """为支持标准 HTTP Range 的官方 ZIP 提供与 OSS 相同的只读接口。"""

    def __init__(self, url):
        self.url = url

    def head_object(self, key):
        result = subprocess.run(["curl", "-fsSI", "--retry", "3", "--max-time", "30", self.url],
                                check=True, capture_output=True)
        lengths = [int(line.split(b":", 1)[1].strip()) for line in result.stdout.splitlines()
                   if line.lower().startswith(b"content-length:")]
        if not lengths:
            raise OSError("dataset server did not provide content length")
        return SimpleNamespace(content_length=lengths[-1])

    def get_object(self, key, byte_range):
        start, end = byte_range
        result = subprocess.run([
            "curl", "-fsSL", "--retry", "3", "--retry-all-errors", "--max-time", "60",
            "--range", f"{start}-{end}", "--max-filesize", str(end - start + 1), self.url,
        ], check=True, capture_output=True)
        return io.BytesIO(result.stdout)


class SharedRangeCache:
    """多个 ZIP reader 共享分块缓存；网络请求在块锁内，避免重复读取同一块。"""

    def __init__(self, bucket, key, block_size=256 * 1024, max_blocks=256):
        self.bucket, self.key = bucket, key
        self.size = bucket.head_object(key).content_length
        self.block_size, self.max_blocks = block_size, max_blocks
        self.cache = OrderedDict()
        self.guard = threading.Lock()
        self.block_locks = {}

    def get_block(self, index):
        with self.guard:
            lock = self.block_locks.setdefault(index, threading.Lock())
        with lock:
            with self.guard:
                if index in self.cache:
                    self.cache.move_to_end(index)
                    return self.cache[index]
            start = index * self.block_size
            end = min(self.size, start + self.block_size) - 1
            data = self.bucket.get_object(self.key, byte_range=(start, end)).read()
            if len(data) != end - start + 1:
                raise OSError("incomplete dataset byte range")
            with self.guard:
                self.cache[index] = data
                while len(self.cache) > self.max_blocks:
                    self.cache.popitem(last=False)
            return data


class RangeReader(io.RawIOBase):
    def __init__(self, cache):
        self.cache, self.position = cache, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("invalid whence")
        target = offset + (0 if whence == 0 else self.position if whence == 1 else self.cache.size)
        if target < 0:
            raise ValueError("negative seek")
        self.position = target
        return target

    def read(self, size=-1):
        end = self.cache.size if size < 0 else min(self.cache.size, self.position + size)
        chunks = []
        while self.position < end:
            index, offset = divmod(self.position, self.cache.block_size)
            block = self.cache.get_block(index)
            length = min(len(block) - offset, end - self.position)
            chunks.append(block[offset:offset + length])
            self.position += length
        return b"".join(chunks)


def public_aishell_zip():
    """遵循 ModelScope SDK 的公开数据访问流程，临时凭证只留在内存中。"""
    import json

    import oss2

    api = "https://modelscope.cn/api/v1/datasets/modelscope/speech_asr_aishell1_testsets"
    # 与字节流传输统一使用 curl 的重试机制，保留 TLS 证书校验。
    response = subprocess.run([
        "curl", "-fsSL", "--retry", "3", "--retry-all-errors", "--max-time", "30",
        api + "/ststoken?Revision=master",
    ], check=True, capture_output=True)
    body = json.loads(response.stdout)
    if body.get("Code") != 200:
        raise RuntimeError("ModelScope public dataset authorization unavailable")
    config = body["Data"]
    auth = oss2.StsAuth(config["AccessId"], config["AccessSecret"], config["SecurityToken"])
    bucket = oss2.Bucket(auth, "https://" + config["Region"] + ".aliyuncs.com", config["Bucket"])
    for prefix in (config["Dir"], config["BackupDir"]):
        key = prefix.rstrip("/") + "/speech_asr_aishell1_testsets.zip"
        if bucket.object_exists(key):
            return SharedRangeCache(bucket, key)
    raise RuntimeError("public dataset archive not found through supported API")

