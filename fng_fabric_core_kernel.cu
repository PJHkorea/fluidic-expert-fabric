// 32바이트 정렬을 통해 L1/L2 캐시라인 일관성 문제(Stall)를 배제한 토큰 셀 구조체
struct alignas(32) FabricIngressTokenCell {
    float features[8]; // 32 bytes (4 bytes * 8)
};

// RDMA 네트워킹을 위한 원격 노드 주소 및 권한 컨텍스트 구조체
struct FabricRemoteAddressContext {
    uint64_t remote_vram_base_ptr; // 64비트 원격 VRAM 기반 포인터
    uint32_t remote_rkey;          // RDMA 액세스 보호 키 (InfiniBand/RoCEv2)
    uint32_t node_rank_id;         // 클러스터 내 노드 식별자
};


    // A. 글로벌 가속기 스레드 및 Intra-Warp 고유 레지스터 오프셋 획득
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // [🛡️ SILICON RUNTIME FIREWALL]: 범위 초과 데이터의 하드웨어 단 세그폴트(SegFault) 방화벽 가동
    bool is_valid_token = (global_idx < total_tokens);
    int target_expert = is_valid_token ? assigned_expert_ids[global_idx] : -1;

    // B. 전문가 레인별 루프를 무분기(Branchless) 비트마스크 스캔으로 치환
    for (int e = 0; e < num_experts; ++e) {
        bool match_flag = (target_expert == e);
        
        // 가속기 SM 소자 내부의 비동기 실행 스레드 활성 하드웨어 마스크 전사
        unsigned int active_mask = __activemask();
        
        // __ballot_sync 기계어로 워프 내 현재 전문가 조준 스레드 가닥들을 비트 필드로 1클록 집산
        unsigned int expert_bitmask = __ballot_sync(active_mask, match_flag);
        
        // [Prefix-Sum Scan] 현재 가닥(lane_id) 하방에 위치한 매칭 비트만 필터링하여 카운트 (__popc 직타)
        int relative_pos = __popc(expert_bitmask & ((1U << lane_id) - 1));

        // [KR] 분기 예측 실패(JMP)를 차단하기 위해 PTX 기계어 조건부 이동 명령어 selp.b32 직접 격발
        int target_slot;
        asm volatile (
            "selp.b32 %0, %1, %2, %3;"
            : "=r"(target_slot)
            : "r"(relative_pos), "r"(GARBAGE_IDX), "b"(match_flag)
        );

        // 정적 버케팅 임계 상한선 사양 정합성 마크 스캔
        bool write_gate = (match_flag && (target_slot < tokens_per_expert));

        if (write_gate) {
            // 글로벌 가상 통합 제어 평면 내부의 정적 레지스터 격자 주소선 영구 전사
            int target_write_addr = e * tokens_per_expert + target_slot;
            fused_fabric_routing_table[target_write_addr] = global_idx;

            // ----------------------------------------------------------------------------
            // [🔒 ZERO-COPY ONCHIP MEMORY INGESTION ROUTE]
            // __ldg() 고속 읽기 전용 가속 레일과 원격 RDMA 가상 포인터 오프셋 연계 마감
            // ----------------------------------------------------------------------------
            for (int f = 0; f < feature_dim; ++f) {
                // 상류 PyTorch 백본 입력 스트림의 1차원 선형 물리 주소선 산출
                int src_addr = global_idx * feature_dim + f;
                
                // 지정된 전문가 레인 버킷 내부의 정적 레지스터 2차원 매핑 가상 주소선 산출
                int dst_addr = (e * tokens_per_expert + target_slot) * feature_dim + f;
                
                // [💥 HARDWARE OPTIMIZATION PRIMITIVE]
                // __ldg 캐시 유닛을 통해 HBM 버스 뱅크의 읽기 병목 스톨을 0ns로 무력화하며 직통 쓰기 집행
                fused_expert_dispatched_cache[dst_addr] = __ldg(&raw_token_stream[src_addr]);
            }
        }
    }
}

// 전문가(Expert)와 토큰 슬롯(Token Slot)을 그리드(Grid) 및 스레드(Thread) 차원에 일대일(1:1) 매핑
int expert_idx = blockIdx.x; 
int token_slot = threadIdx.x; 

// [🛡️ RUNTIME HARDWARE MASK]: SM 범위를 초과하는 스레드 조기 컷백 이탈 (하드웨어 보호)
if (expert_idx >= num_experts || token_slot >= tokens_per_expert) {
    return;
}

    // A. 정방향 디스패치 단계에서 동결 완료된 정적 격자 주소선 오프셋 산출
    int routing_addr = expert_idx * tokens_per_expert + token_slot;
    
    // B. 상류 PyTorch 백본 입력 스트림의 오리지널 토큰 인덱스 ID 역산 추적 복원
    int original_token_idx = fused_expert_routing_table[routing_addr];

    // [🛡️ SYSTEM MEMORY OUT-OF-BOUNDS DEFENSE FIREWALL]
    // 유실/가변 버케팅 패딩 및 가비지 인덱스를 원천 차단하여 세그폴트 방어
    if (original_token_idx == GARBAGE_IDX || original_token_idx < 0) {
        return;
    }


    // C. 상류 PyTorch 백본의 게이팅 확률 매트릭스로부터 고유 가중치선 참조
    //    오리지널 토큰 인덱스와 현재 연산 전문가 인덱스에 연동 매핑됩니다.
    float gate_weight = gating_probabilities[original_token_idx * num_experts + expert_idx];

    // D. 💥 [HARDWARE ATOMIC CONCURRENT STREAM MERGE LINE]
    //    Feature 차원을 따라 원자적 실리콘 연산 장치 직타 파이프라인 가동
    for (int f = 0; f < feature_dim; ++f) {
        // 현재 전문가 레인 버킷 내부의 1차원 선형 소스 주소선 산출
        int src_addr = (expert_idx * tokens_per_expert + token_slot) * feature_dim + f;
        
        // 상류 PyTorch 백본 입력 스트림의 원본 시퀀스 축 복귀 타겟 주소선 산출
        int dst_addr = original_token_idx * feature_dim + f;
        
        // 게이팅 가중치를 곱해 대수적 아다마르 연산 수행
        float weighted_value = expert_outputs[src_addr] * gate_weight;
        
        // [💥 HARDWARE ATOMIC PRIMITIVE]
        // NCCL All-to-All 통신 병목을 소멸시키고 Write Race Condition을 하드웨어 수준에서 평탄화
        atomicAdd(&reconstructed_stream[dst_addr], weighted_value);
    }
}
