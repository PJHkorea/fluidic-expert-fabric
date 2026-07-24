# ====================================================================
# [PIM-HBM ZERO-COPY HARDWARE MoE CORE INFRASTRUCTURE - V1.0]
# @file: fng_fabric_autograd_bridge.py
# ====================================================================

import torch
import jax
import jax.numpy as jnp
from torch.utils.dlpack import to_dlpack, from_dlpack
from jax.dlpack import to_dlpack as jax_to_dlpack
from jax.dlpack import from_dlpack as jax_from_dlpack
from typing import Tuple, Any

# 상위 글로벌 네트워크 패브릭 사양서 상수 연동 상속
from fng_fabric_config import NUM_EXPERTS, FEATURE_DIM

class FngFabricAutogradBridgeFunction(torch.autograd.Function):
    """
    [GLOBAL FABRIC MULTI-NODE INTERLOCK FUNCTION]
    파이토치 C++ Autograd 시스템 내부에 JAX/XLA SPMD 분산 VJP 연산 장치를
    0바이트 복사 프로토콜(DLPack Dual-Pointer Hijacking)로 주입하는 양방향 가상화 교량입니다.
    """
    
    @staticmethod
    def forward(
        ctx: Any, 
        hidden_states: torch.Tensor, 
        gate_logits: torch.Tensor, 
        sharding_tower: Any, 
        mesh: Any,
        bucket_size: int,
        tokens_per_expert: int
    ) -> torch.Tensor:
        """
        [📢 FORWARD DISPATCH INTERLOCK]: 파이토치 VRAM 물리 기저 주소를 JAX 분산 버스로 수입
        """
        # [🛡️ HARDWARE CONTIGUITY DEFENSE]: 분산 메모리 정렬의 연속성이 깨져 발생하는 수치 폭주 선제 차단
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        if not gate_logits.is_contiguous():
            gate_logits = gate_logits.contiguous()

        # 역방향 오차 전파 단열 터널 구동용 파라미터 컨텍스트 박제
        ctx.sharding_tower = sharding_tower
        ctx.mesh = mesh
        ctx.bucket_size = bucket_size
        ctx.tokens_per_expert = tokens_per_expert

        # [🔒 0-COPY INTER-FRAMEWORK INGESTION]: DLPack 표준 바인딩을 통해 복사 레이턴시 영점화
        # 파이토치가 소유한 가속기 물리 주소선을 JAX 분산 디바이스 어레이 변수로 무복사 직통 전사합니다.
        jax_tokens = jax_from_dlpack(to_dlpack(hidden_states))
        jax_logits = jax_from_dlpack(to_dlpack(gate_logits))

        # [🌀 DISTRIBUTED JAX VJP ENGAGEMENT]: 정방향 출력 사출과 동시에 역방향 미분 기계어 주소선(_fabric_vjp_fn) 포획
        with mesh:
            jax_outputs, fabric_vjp_fn = jax.vjp(
                lambda h, g: sharding_tower.parallel_fabric_dispatch_routing(h, g, bucket_size, tokens_per_expert),
                jax_tokens,
                jax_logits
            )
            
        # [🔒 EXTENDED LIFE-CYCLE GUARD]: 비동기 가비지 컬렉터(GC)의 조기 주소 파손을 방어하기 위한 레지스터 홀딩
        ctx.fabric_vjp_fn = fabric_vjp_fn
        ctx.save_for_backward(hidden_states, gate_logits)

        # 정제 완결된 JAX 글로벌 분산 아웃풋 다양체를 다시 파이토치 VRAM 공간으로 0바이트 복사 회수 토출
        torch_outputs = from_dlpack(jax_to_dlpack(jax_outputs))
        return torch_outputs


    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None, None, None]:
        """
        [📢 BACKWARD PARALLEL COMBINE]: 단열 백프로파게이션 터널(Adiabatic Backpropagation Tunnel) 가동
        상류 파이토치 Autograd 미분 체인과 하방 JAX SPMD VJP 주소선을 무복사 직결합니다.
        """
        # [🛡️ HARDWARE CONTIGUITY DEFENSE]: 오차 전파 벡터 행렬의 물리 정렬 연속성 보존
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        # 정방향 패스 통과 시점에 컨텍스트(ctx) 내부에 영구 박제해 둔 XLA VJP 기계어 주소선 및 토폴로지 사양 로드
        fabric_vjp_fn = ctx.fabric_vjp_fn
        sharding_tower = ctx.sharding_tower
        mesh = ctx.mesh
        bucket_size = ctx.bucket_size
        tokens_per_expert = ctx.tokens_per_expert
        
        # [🔒 ZERO-COPY POINTER HIJACKING]: 인입된 상류 파이토치 오차 행렬의 가속기 주소선을 DLPack으로 하이재킹
        jax_grad_output = jax_from_dlpack(to_dlpack(grad_output))

        # [💥 HARDWARE NATIVE ATOMIC INTENSIVE RUNTIME]
        # XLA VJP 역산 관류를 기폭하여, 하방 C++ 커널의 atomicAdd() 실리콘 기계어 트랙 위로 그라디언트를 조준 사격합니다.
        with mesh:
            # 샤딩 타워의 거울 대칭형 parallel_fabric_combine_routing 연산 유닛과 상호 융합 인터록
            grad_hidden, grad_logits = fabric_vjp_fn(
                sharding_tower.parallel_fabric_combine_routing(
                    jax_grad_output, 
                    jax_from_dlpack(to_dlpack(ctx.saved_tensors[1])), # gate_logits 토큰 매핑 데이터 복원 수입
                    bucket_size, 
                    tokens_per_expert
                )
            )

        # [🛡️ LINEAR INTERCONNECT ALIGNMENT FENCE]
        # 미분 대상이 아닌 인자 축(sharding_tower, mesh, bucket_size, tokens_per_expert)의 사양 서열에 정확히 대응하여
        # 파이토치 C++ Autograd 엔진이 요구하는 명시적 'None' 패딩 반환 마킹 규격을 수호합니다.
        torch_grad_hidden = from_dlpack(jax_to_dlpack(grad_hidden))
        torch_grad_logits = from_dlpack(jax_to_dlpack(grad_logits))

        return torch_grad_hidden, torch_grad_logits, None, None, None, None


class FngFabricAutogradBridge:
    """
    [HIGH-LEVEL CO-DESIGN FABRIC INTERFACE]
    실제 상용 LLM 백본 모델(Mixtral / DeepSeek-V3)의 런타임 몽키 패치 주입 단에서
    가상 MUX 제어 평면을 손쉽게 트리거할 수 있도록 직관적인 호출 규격을 제공하는 인프라 래퍼입니다.
    """
    def __init__(self, sharding_tower: Any, mesh: Any, bucket_size: int, tokens_per_expert: int):
        """
        다중 노드 거시 관제탑 및 가속기 토폴오지 토큰 슬롯 사양 상속 앵커링
        """
        self.sharding_tower = sharding_tower
        self.mesh = mesh
        self.bucket_size = bucket_size
        self.tokens_per_expert = tokens_per_expert

    def __call__(self, hidden_states: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        [⚡ INLINE INFRASTRUCTURE GATEWAY]
        상류 파이토치 forward 루프가 인입되는 순간, 하부 0-Copy 단열 자동미분 터널을 기폭합니다.
        """
        return FngFabricAutogradBridgeFunction.apply(
            hidden_states, 
            gate_logits, 
            self.sharding_tower, 
            self.mesh,
            self.bucket_size,
            self.tokens_per_expert
        )


print("====================================================================")
print("🔄 DUAL-FRAMEWORK DISTRIBUTED AUTOGRAD-VJP BRIDGE COMPLETELY MOUNTED")
print("   ├─ [PROTOCAL] Non-Copy Cross-Framework Device Address Handover Active.")
print("   └─ [TUNNELING] Multi-Node Symmetrical Error Reflection Sealed.")
print("====================================================================")
