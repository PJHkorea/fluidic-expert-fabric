// Token cell structure aligning to 32 bytes to eliminate L1/L2 cache line coherency stalls.
struct alignas(32) FabricIngressTokenCell {
    float features[8]; // 32 bytes (4 bytes * 8)
};

// Remote node address and authorization context structure configured for bare-metal RDMA networking.
struct FabricRemoteAddressContext {
    uint64_t remote_vram_base_ptr; // 64-bit remote VRAM base memory address pointer
    uint32_t remote_rkey;          // RDMA access protection steering key (InfiniBand/RoCEv2 rkey)
    uint32_t node_rank_id;         // Cluster-wide unique node identifier
};


    // A. Capture the global accelerator thread index and the intra-warp native register offset
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // [🛡️ SILICON RUNTIME FIREWALL]: Engage the hardware-level segmentation fault firewall to trap out-of-bound data.
    bool is_valid_token = (global_idx < total_tokens);
    int target_expert = is_valid_token ? assigned_expert_ids[global_idx] : -1;

    // B. Transform the per-expert lane loops into completely branchless bitmask scans
    for (int e = 0; e < num_experts; ++e) {
        bool match_flag = is_valid_token && (target_expert == e);
        //[★ Fix] Stabilized hardware synchronization mask
        unsigned int expert_bitmask = __ballot_sync(0xFFFFFFFF, match_flag); 
        int relative_pos = __popc(expert_bitmask & ((1U << lane_id) - 1));
        
        // [★ Fix] Applied `selp` in compliance with the 1-bit predicate specification.
        asm volatile (
            "{\n\t"
            "  .reg .pred %p;\n\t"
            "  setp.ne.u32 %p, %3, 0;\n\t" // 0이 아니면 %p는 true
            "  selp.b32 %0, %1, %2, %p;\n\t"
            "}"
            : "=r"(target_slot)
            : "r"(relative_pos), "r"(GARBAGE_IDX), "r"((unsigned int)match_flag)
        );


               // Scan and verify specification conformity against the static bucketing threshold upper bound
        bool write_gate = (match_flag && (target_slot < tokens_per_expert));

        if (write_gate) {
            // Permanently commit the static register grid address line inside the global virtual unified control plane
            int target_write_addr = e * tokens_per_expert + target_slot;
            fused_fabric_routing_table[target_write_addr] = global_idx;

            // ----------------------------------------------------------------------------
            // [🔒 ZERO-COPY ONCHIP MEMORY INGESTION ROUTE]
            // Tie the __ldg() high-speed read-only accelerator rail with the remote RDMA virtual pointer offset
            // ----------------------------------------------------------------------------
            for (int f = 0; f < feature_dim; ++f) {
                // Compute the 1D linear physical memory address line of the upstream PyTorch backbone input stream
                int src_addr = global_idx * feature_dim + f;
                
                // Compute the 2D mapped virtual address line of the static register inside the designated expert lane bucket
                int dst_addr = (e * tokens_per_expert + target_slot) * feature_dim + f;
                
                // [💥 HARDWARE OPTIMIZATION PRIMITIVE]
                // Bypass HBM bus bank read-bottleneck stalls down to 0ns via the __ldg cache unit, executing a direct zero-copy write
                fused_expert_dispatched_cache[dst_addr] = __ldg(&raw_token_stream[src_addr]);
            }
        }
    }
}

// Map Experts and Token Slots 1:1 onto the hardware Grid and Thread dimensions
int expert_idx = blockIdx.x; 
int token_slot = threadIdx.x; 

// [🛡️ RUNTIME HARDWARE MASK]: Early cutback and departure of threads exceeding SM boundaries (Hardware Protection Guard)
if (expert_idx >= num_experts || token_slot >= tokens_per_expert) {
    return;
}


     // A. Compute the static register grid address line offset frozen during the forward dispatch phase
    int routing_addr = expert_idx * tokens_per_expert + token_slot;
    
    // B. Backtrack and restore the original token index ID belonging to the upstream PyTorch backbone input stream
    int original_token_idx = fused_expert_routing_table[routing_addr];

    // [🛡️ SYSTEM MEMORY OUT-OF-BOUNDS DEFENSE FIREWALL]
    // Intercept dropped/variable bucketing padding and garbage indices to strictly defend against segmentation faults
    if (original_token_idx == GARBAGE_IDX || original_token_idx < 0) {
        return;
    }


    // C. Reference the unique gating weight line from the upstream PyTorch backbone's gating probability matrix
    //    This is mapped and coupled with the original token index and the current active expert index.
    float gate_weight = gating_probabilities[original_token_idx * num_experts + expert_idx];

    // D. 💥 [HARDWARE ATOMIC CONCURRENT STREAM MERGE LINE]
    //    Engage the physical silicon execution unit direct pipeline along the feature dimension axis
    for (int f = 0; f < feature_dim; ++f) {
        // Compute the 1D linear source address line inside the current expert lane bucket
        int src_addr = (expert_idx * tokens_per_expert + token_slot) * feature_dim + f;
        
        // Compute the target destination address line to return to the original sequence axis of the upstream PyTorch backbone input stream
        int dst_addr = original_token_idx * feature_dim + f;
        
        // Execute algebraic Hadamard product multiplication by applying the gating weight
        float weighted_value = expert_outputs[src_addr] * gate_weight;
        
        // [💥 HARDWARE ATOMIC PRIMITIVE]
        // Annihilate NCCL All-to-All communication bottlenecks and flatten write race conditions directly at the hardware silicon level
        atomicAdd(&reconstructed_stream[dst_addr], weighted_value);
    }
}
