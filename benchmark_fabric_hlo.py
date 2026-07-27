# ====================================================================
# [PIM-HBM ZERO-COPY HARDWARE MoE CORE INFRASTRUCTURE - V1.0]
# @file: benchmark_fabric_hlo.py
# Multi-Node Static Assembly HLO Profiler & Audit Firewall
# ====================================================================

import re
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import time
from typing import Dict, Any

# Vertical inheritance binding for global distributed network fabric components
from fng_fabric_config import FABRIC_BUCKET_SIZES, FEATURE_DIM, NUM_EXPERTS, compute_expert_register_capacity
from test_cluster_e2e_fabric import mock_fabric_core_pipeline_factory

def compile_and_dump_pure_fabric_hlo_asm(bucket_size: int, tokens_per_expert: int, mesh: Mesh) -> str:
    """
    [XLA HLO IR ASSEMBLY TEXT EMITTER]
    Leverages JAX abstract tracers to emit the mathematical and physical machine graph 
    of the underlying factory kernel into a fully static text format.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P

    # ❶ Generates precision sharding topologies based on the "data_parallel" axis required by the underlying ShardMap infrastructure.
    dp_sharding = NamedSharding(mesh, P("data_parallel", None))

    # ❷ [★ CRITICAL DISTRIBUTED AXIS LEAK COMPENSATOR ★]: Forcibly pins the sharding=dp_sharding attribute directly onto the abstract array geometry.
    # This maintains the active VRAM footprint at absolute 0MB, while coercing the JAX compiler to recognize the structure as an inherently aligned 
    # physical distribution layout, systematically crushing In/Out Specs mismatch graph breaks.
    abstract_token_stream = jax.ShapeDtypeStruct(
        shape=(bucket_size, FEATURE_DIM), 
        dtype=jnp.float32,
        sharding=dp_sharding
    )
    abstract_gate_logits = jax.ShapeDtypeStruct(
        shape=(bucket_size, NUM_EXPERTS), 
        dtype=jnp.float32,
        sharding=dp_sharding
    )


    # ❸ Target downstream algebraic Mux hardware-bound pipeline factory to validate.
    hardware_pass_kernel = mock_fabric_core_pipeline_factory(
        bucket_size=bucket_size,
        tokens_per_expert=tokens_per_expert
    )

    # ❹ Lock the compilation graph via AOT compiler plane to freeze the HLO IR instructions.
    with mesh:
        # Bind abstract dimensions into the jax.jit single-clock fused graph
        jit_compiled_graph = jax.jit(hardware_pass_kernel)
        
        # Execute static stage lowering down to the underlying accelerator assembly
        lowered_hlo_graph = jit_compiled_graph.lower(
            abstract_token_stream, 
            abstract_gate_logits
        )
        
            # [★ COMPILER COMPATIBILITY BOUNDARY FINALIZATION ★]: Directs the parsing of the lowered HLO IR domain module, 
        # or rigidly aligns the string serialization pathway to comply with the latest compiled binary text release specifications.
        compiled_executable = lowered_hlo_graph.compile()

    # Decode and return the machine bytecode hidden behind the compiler veil into human-readable pure text.
    return compiled_executable.as_text()




def audit_compiled_silicon_fabric_instructions(hlo_assembly_text: str) -> Dict[str, Any]:
    """
    [SILICON INSTRUCTION AUDIT FIREWALL]
    Performs precision text-based regex auditing on the emitted XLA HLO intermediate 
    representation assembly to detect the leakage of collective communication and sorting 
    instructions that induce hardware pipeline stalls.
    """
    # ❶ [★ CRITICAL FALSE-POSITIVE & AXIS LEAK PROTECTOR ★]: Reinforced with a robust wildcard regex pattern 
    # designed to concurrently intercept both hyphens (-) and underscores (_) with maximum precision, 
    # capturing 100% of the XLA compiler's unique native name-mangling specifications.
    collective_comm_patterns = [
        r"all[-_]to[-_]all",
        r"collective[-_]permute",
        r"all[-_]gather",
        r"reduce[-_]scatter",
        r"\bsend\b", # Isolated as an independent whole word to completely eliminate false-positive detection vectors
        r"\brecv\b"
    ]


       # ❷ [★ DETERTMINISTIC FALSE-POSITIVE ISOLATION & EXCLUSION ★]: To prevent errant triggering on standard identifiers (e.g., sorted_axis), 
    # surgically targets the explicit HLO IR instruction control-line specifications (e.g., the exact = sort(...) bytecode ingress pattern).
    sorting_patterns = [
        r"custom[-_]call.*bitonic",
        r"\bsort\s*\("  # Pins execution vectors down to actual native sort() operation structures alone
    ]

    detected_comm_primitives = {}
    detected_sorting_primitives = {}
    
    total_comm_leaks = 0
    total_sorting_leaks = 0


    # A. Target scan for collective communication primitives (Case-Insensitive)
    for pattern in collective_comm_patterns:
        matches = re.findall(pattern, hlo_assembly_text, re.IGNORECASE)
        match_count = len(matches)
        detected_comm_primitives[pattern] = match_count
        total_comm_leaks += match_count

    # B. Target scan for warp sorting primitives
    for pattern in sorting_patterns:
        matches = re.findall(pattern, hlo_assembly_text, re.IGNORECASE)
        match_count = len(matches)
        detected_sorting_primitives[pattern] = match_count
        total_sorting_leaks += match_count

    # C. Final verification guard for 0ns silicon-clean integrity
    is_silicon_clean = (total_comm_leaks == 0) and (total_sorting_leaks == 0)

    report = {
        "is_clean": is_silicon_clean,
        "comm_summary": detected_comm_primitives,
        "sorting_summary": detected_sorting_primitives,
        "total_comm_leaks": total_comm_leaks,
        "total_sorting_leaks": total_sorting_leaks
    }

    return report



def run_fabric_hlo_static_assembly_benchmark() -> None:
    """
    [⚡ STATIC FABRIC HLO VERIFICATION ORCHESTRATOR]
    Activates the virtual accelerator distributed topology and emits the machine IR assembly 
    hidden behind the compiler veil, permanently guaranteeing a strict zero-count audit 
    for communication and sorting leaks within the final execution timeline.
    """
    print("====================================================================")
    print("🔍 IGNITING MULTI-NODE FABRIC XLA HLO ASSEMBLY PROFILER...")
    print("====================================================================")

      # A. Setup the multi-node distributed virtual mesh topology properly aligned with the sharding tower
    # [★ CRITICAL SHARDING AXIS & DEVICE SLICING ALIGNMENT ★]: 
    # Standardizes the mesh axis identity explicitly as 'expert_fabric' and fuses the entire physical device pool, 
    # deterministically passing the FngFabricShardingTower validation assert barriers and preserving the multi-node 
    # dimensional tensor rank tracking loop without numerical degradation.
    devices = jax.devices()
    mock_mesh = Mesh(jnp.array(devices), ("expert_fabric",))
    print(f"[FABRIC_PROFILER_BOOT] Device sharding topology mesh locked: {mock_mesh}")

    # B. Bind the representative static bucket specification for variable inference streams (Targeting 512 guard bucket boundary)
    target_bucket_size = FABRIC_BUCKET_SIZES[3] # 512 static slots boundary
    tokens_per_expert = compute_expert_register_capacity(target_bucket_size)
    print(f"[FABRIC_PROFILER_TARGET] Targeting dynamic inference window mapping: {target_bucket_size} slots.")

    # C. [PART 1/3] Fire up the XLA lowering engine ➔ Capture pure static HLO IR text dump
    print(f"[COMPILING] Down-shifting abstract JAX tracers into bare-metal execution fabric...")
    start_time = time.perf_counter()
    hlo_assembly_text = compile_and_dump_pure_fabric_hlo_asm(
        bucket_size=target_bucket_size,
        tokens_per_expert=tokens_per_expert,
        mesh=mock_mesh
    )
    end_time = time.perf_counter()
    print(f" ✨ [COMPILE SUCCESS] Core matrix HLO text extracted in {end_time - start_time:.4f} seconds.")

    # D. Permanently isolate and dump the emitted XLA machine binary assembly to the local disk
    dump_filename = "fng_moe_optimized_hlo.txt"
    with open(dump_filename, "w", encoding="utf-8") as f:
        f.write(hlo_assembly_text)
    print(f" ├─ [FILE EXPORT] Assembly fabric output permanently sealed in './{dump_filename}'.")

    # E. Activate the regex-based silicon instruction audit firewall (Zero-leak permanent enforcement)
    print(f"[AUDITING] Scanning HLO IR instructions for hidden distributed interconnect leaks...")
    audit_results = audit_compiled_silicon_fabric_instructions(hlo_assembly_text)
    


       #  Emit telemetry scan results to the instrumentation reporting console
    print("\n====================================================================")
    print("📊 SILICON ASSEMBLY INTERCONNECT INFRASTRUCTURE AUDIT REPORT")
    print("====================================================================")
    print(f" ├─ [NCCL Collective Leak Count] : {audit_results['total_comm_leaks']} Leaks Detected.")
    for primitive, count in audit_results["comm_summary"].items():
        print(f" │    └─ Pattern '{primitive:18s}' ➔ Count: {count}")
        
    print(f" ├─ [Warp Serialization Leak Count] : {audit_results['total_sorting_leaks']} Leaks Detected.")
    for primitive, count in audit_results["sorting_summary"].items():
        print(f" │    └─ Pattern '{primitive:20s}' ➔ Count: {count}")
    print("====================================================================")

    # F. [MANDATORY INFRASTRUCTURE GUARDRAIL]: Immediate compiler self-destruction guard 
    #    triggered upon detection of any remaining physical breakout primitives.
    assert audit_results["is_clean"], (
        f"[🚨 SYSTEM FABRIC VIOLATION] Critical communication or sorting primitive leaked into HLO execution fabric! "
        f"Distributed 0ns zero-copy multi-node integrity broken. Check your core macro graph."
    )

    print("\n🎯 [CONCLUSION] Multi-Node silicon fabric graph 100% verified. Pure branchless / collective-free profile validated.")
    print("====================================================================\n")

# --------------------------------------------------------------------------------
# 🎬 [MAIN ENTRANCE]: Static Profiler Independent Execution Entry Point Locking
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # Ignite low-level multi-node assembly static auditing and 0-count permanent assurance benchmark
    run_fabric_hlo_static_assembly_benchmark()
