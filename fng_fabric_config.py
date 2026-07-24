import os
from typing import Final, Tuple

# JAX 메모리 선점 방지 및 플랫폼 할당자 설정
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# RDMA 통신 지연 은닉 및 XLA 최적화 플래그 고정
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_graph_level=3 "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_all_gather_combine_threshold_bytes=134217728 "
    "--xla_gpu_reduce_scatter_combine_threshold_bytes=134217728"
)

# RoCEv2 기반 토폴로지 및 메모리 구성용 하드코딩된 상수
NUM_EXPERTS: Final[int] = 8
FEATURE_DIM: Final[int] = 4096
FABRIC_SPARE_RATIO: Final[float] = 0.05
FABRIC_ALIGNMENT_BYTES: Final[int] = 32

# 다중 노드 추론 시 Re-compilation 방지를 위한 버킷 사이즈 설정
FABRIC_BUCKET_SIZES: Final[Tuple[int, ...]] = (64, 128, 256, 512, 1024, 2048)

def compute_expert_register_capacity(bucket_size: int) -> int:
    """O(1) 추론 복잡도를 위한 레지스터 용량 계산"""
    return bucket_size
