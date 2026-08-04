# fluidic-expert-fabric (PoC)

This repository contains an exploratory Proof of Concept (PoC) investigating a hardware-software co-design approach for distributed Mixture-of-Experts (MoE) architectures, such as DeepSeek-V3 and Mixtral-8x7B. This project represents a modest attempt to re-examine traditional interconnect bandwidth limitations by directly bridging remote RDMA (RoCEv2) virtual address lines with multi-axis jax.shard_map and jax.sharding.NamedSharding structures.

By experimenting with 64-bit virtual memory address pointer mapping across multi-node environments, we hope to investigate methods for minimizing legacy inter-node data replication (`memcpy`) overheads, mitigating NCCL All-to-All communication stalls, and safely preserving numerical homeostasis during distributed parameter dispatch and gradient accumulation stages without introducing dynamic allocation bubbles.


---

## 🌊 Low-Layer Hardware Co-Design Infrastructure Suite

This ecosystem represents an ongoing exploratory attempt to examine methods for mitigating physical constraints surrounding heterogeneous framework runtimes, accelerator memory interconnect bandwidth boundaries, and potential HBM memory subsystem stalls.

### 🔌 Component Ecosystem & Sovereign Repositories

*   **`Fluidic_Network_Grid` (FNG):** A preliminary attempt to establish a macro-level traffic bus and tensor virtualization bridge, aiming to orchestrate zero-copy data routing across JAX SPMD sharding architectures and PyTorch autograd lanes via safe DLPack reference pinning.
*   **`pim-hbm-bypass`:** An exploratory research effort evaluating PIM bank activation patterns and CUDA memory virtualization primitives, attempting to inspect fault-tolerance behaviors and dynamic rank adjustments under simulated hardware degradation states.
*   **`pim-moe-core`:** A low-level branchless MUX core kernel and automatic differentiation bridge, experimenting with inline `selp` instructions and atomic hardware paths to investigate the reduction of local framework wrapping overheads within Sparse MoE paths.
*   **`fluidic-expert-fabric`:** A macro-scale consolidation of our infrastructure experiments, attempting to scale virtual address-stride swapping onto simulated RDMA (RoCEv2) inter-node fabric lines to gently examine the mitigation of NCCL All-to-All communication stalls under dynamic sequence layouts.



---

## 📐 Key Areas of Exploration & Technical Approaches

### 1. Multi-Node Virtual Address Mapping Experiments
* **Aims to mitigate token chunk serialization** and evaluate bottlenecks traditionally associated with physical interconnect cable transits between nodes.
* **Attempts to evaluate a globally unified memory topology view** across distinct simulated hardware chassis configurations.
* **Experiments with local address tracking** to inspect data proximity factors the moment gating targets are calculated.

### 2. Framework Pointer Interlocking Protocols
* **Gently bridges high-level Python tensors** with low-level memory descriptors via standard framework serialization boundaries.
* **Implements an asynchronous cross-stream memory-map protocol** designed to route reduction tasks straight toward native atomic accumulation circuits (`unique_indices=False`).
* **Attempts to reduce memory allocation overhead bubbles** during runtime execution loops by utilizing strict reference preservation techniques.

### 3. Static Bucket Power-of-2 Bounds
* **Aims to protect the XLA tracer graph from recompilation spikes** caused by dynamic input shape fluctuations in multi-node execution streams.
* **Pre-compiles and fixes memory bucket boundaries offline** (e.g., $64 \rightarrow 128 \rightarrow 256 \rightarrow \dots$) to encourage smooth pre-allocation layouts.
* **Investigates methods for encouraging deterministic execution** times regardless of fluidic token distribution shifts across experts.

### 4. Macro-Topology Reserve Masking Trials
* **Consolidates auxiliary backup bank buffers** across the distributed computing cluster layout.
* **Executes conditional address masking via algebraic primitives** (such as `jnp.where` and shape-aligned `jnp.take_along_axis`) during simulated degradation states.
* **Investigates failover routing pathways** to observe execution continuity without triggering unexpected compiler cache re-evaluation loops.

---

## 🧬 Architectural Repository Map & Component Outlines

The file structure is organized into a sundered layout, separating macro-level sharding orchestrators from low-level register selection kernels.

```text
fluidic-expert-fabric/
├── benchmark_fabric_hlo.py       # Static HLO assembly profiler attempting to evaluate 0-count collective communication paths
├── fng_fabric_config.py          # Global distributed network descriptors & environment memory alignment rules
├── fng_fabric_core_kernel.cu     # Bare-metal CUDA kernels experimenting with warp-level bitmask MUX execution paths
├── fng_fabric_sharding_tower.py  # Macro-topology manager investigating multi-node abstract virtual array allocations
├── fng_fabric_autograd_bridge.py # Cross-framework reference pinning bridge linking PyTorch Autograd & JAX VJP timelines
├── fng_fabric_dynamic_adapter.py # Pre-compiler bucket matching layer attempting to mitigate dynamic JIT tracer stalls
├── fng_fabric_monkey_patch.py    # Runtime dynamic injection factory evaluating Mixtral and DeepSeek forward hook points
└── test_cluster_e2e_fabric.py    # Integrated test framework evaluating sequence variations under simulated failure boundaries
```


### 🧬 Detailed File Breakdowns & Research Scope

*   **`benchmark_fabric_hlo.py`:**  
    An exploratory intermediate representation profiler utilizing abstract JAX tracers to inspect compiled XLA HLO text graphs. It attempts to scan for unexpected collective communication remnants via automated wildcard pattern evaluation to observe a collective-free profile.

*   **`fng_fabric_config.py`:**  
    Handles global network environmental descriptors and memory alignment parameters. It defines the experimental structural layout for cluster-wide virtual address mapping and seals conservative pre-set compiler optimization flags.

*   **`fng_fabric_core_kernel.cu`:**  
    Contains bare-metal C++/CUDA communication kernels experimenting with warp-synchronous bitwise operations. It utilizes explicit `0xFFFFFFFF` synchronization masks to investigate deadlocks, alongside inline `selp.b32` assembly circuits for conditional routing.

*   **`fng_fabric_sharding_tower.py`:**  
    A macro-topology manager responsible for testing virtual device arrays via `alloc_abstract_array` primitives. It aims to observe global memory matrix views via JAX/XLA SPMD `shard_map` structures without triggering dynamic size unrolling.

*   **`fng_fabric_autograd_bridge.py`:**  
    An interlock function encapsulating PyTorch C++ Autograd timelines and JAX VJP multi-node derivative paths. It implements explicit DLPack reference pinning and cross-stream `record_stream` barriers to investigate asynchronous lifecycle isolation.

*   **`fng_fabric_dynamic_adapter.py`:**  
    A static pre-compiler module that maps runtime tokens to fixed offline bucketing levels ($64 \rightarrow 128 \rightarrow 256 \rightarrow \dots$). It applies an algebraic vacuum masking sentinel (`-1e9`) to protect the downstream execution graph from tracer serialization delays.

*   **`fng_fabric_monkey_patch.py`:**  
    A dynamic CPython hook factory designed to intercept commercial MoE block definitions at runtime. It incorporates internal `_fng_fabric_patched` guard gates and `clear_compilation_cache()` invocations to experiment with clean graph swapping.

*   **`test_cluster_e2e_fabric.py`:**  
    An integrated infrastructure simulation layer configured to profile numerical convergence behavior, adiabatic automatic differentiation paths, and conditional fallback masks under dynamic sequence distribution shifts.


---

## 📐 Exploratory Mathematical Formulation & Execution Manifold

To investigate the mitigation of Host-to-Device and Inter-Node JMP prediction stalls, this PoC experiments with a branchless approach to local token allocation mapping, attempting to realign execution flows via synchronized algebraic scan primitives:

### 1. Warp-Synchronous Bitmask MUX Formulation
Rather than relying on iterative host-side loop branches, the underlying CUDA kernel attempts to evaluate alignment offsets inside single-clock warp instructions, experimenting with binary indicator conditions ($\mathbb{I}$) combined with bitwise population counts:

$$\mathcal{M}_{e, t} = \mathbb{I}\big( \text{argmax}(\mathbf{g}_{t}) == e \big)$$

$$\mathcal{P}_{e, t} = \left( \sum_{k=1}^{t} \mathcal{M}_{e, k} \right) - 1$$

Where $\mathbf{g}_{t}$ represents the gating logit vector for token $t$, $e$ denotes the target expert lane, and $\mathcal{P}_{e, t}$ yields the relative sequence coordinate offset within the buffer matrix. By mapping this logic to inline `selp` assembly circuits, we hope to inspect register routing behavior with minimal execution stalls.

### 2. Adiabatic Gradient Combine (Concurrent Write Exploration)
During the backward error propagation pass, multiple distributed expert shards are coupled symmetrically back into the original sequence topology. To evaluate write race mitigation and minimize standard runtime synchronization barriers, the design attempts to map the gradient accumulation directly onto native streaming multiprocessor memory controller lanes:

$$\mathbf{\nabla}_{\mathbf{x}_{t}} \mathcal{L} = \sum_{e=1}^{E} \mathcal{M}_{e, t} \cdot \sigma(\mathbf{g}_{t})_{e} \cdot \left[ \text{atomicAdd}\left( \mathbf{\nabla}_{\mathbf{y}_{e, \mathcal{P}_{e, t}}} \mathcal{L} \right) \right]$$

This mathematical mapping is integrated to test whether our JAX SPMD VJP pipeline can be bound to the hardware's concurrent write capability via the `unique_indices=False` parameter, gently investigating an automatic differentiation tunnel with reduced reference leakage.


---

## 📊 Static Assembly Telemetry & HLO Audit Profiles

This repository incorporates an exploratory pre-compilation auditing utility (`benchmark_fabric_hlo.py`) designed to statically inspect the generated High-Level Optimizer (HLO) IR machine-code text before model deployment. By scanning the compiled executable bytecode via canonical wildcard regex mapping, this script attempts to examine whether any unexpected collective communication remnants or sequential sorting instructions have leaked into the final deployment pipeline, gently supporting a verified, collective-free execution profile.

---

## 🚀 Usage Example & Experimental Integration

The following minimalistic setup illustrates how the infrastructure hooks can be gently introduced into a model compilation plane. We are continuously optimizing this interface to minimize host-side intervention and avoid dynamic graph breaks:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import torch
from transformers import AutoModelForCausalLM

from fluidic_expert_fabric import (
    FngFabricShardingTower,
    FngFabricDynamicShapeAdapter,
    inject_fng_fabric_infrastructure_hook
)

# 1. Establish the multi-axis distributed virtual mesh topology
devices = jax.devices()
# Generates data_parallel and expert_fabric axes to align with static HLO tracing blueprints
global_fabric_mesh = Mesh(jnp.array(devices).reshape(8, -1), ("data_parallel", "expert_fabric"))

# 2. Initialize the distributed macro sharding control tower
fabric_tower = FngFabricShardingTower(mesh=global_fabric_mesh)

# 3. Initialize the static compiler adapter to inspect dynamic shape isolation
fabric_adapter = FngFabricDynamicShapeAdapter(sharding_tower=fabric_tower, mesh=global_fabric_mesh)

# 4. Load the commercial-grade parameter model (e.g., DeepSeek-V3)
# Please ensure your node cluster possesses sufficient VRAM capacity before execution.
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V3", 
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# 5. Ingest the zero-copy infrastructure hook into the running compilation plane
# Once executed, the core attention/expert layers will attempt an execution-fusion trial.
model = inject_fng_fabric_infrastructure_hook(model, fabric_adapter)

# Multi-node forward and backward routing can now be monitored via the exploratory fabric.
```


## 📜 License

```text
Copyright (c) 2026 PJHkorea. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
```

- This repository contains open-source code distributed under the **Apache License 2.0**. The full license text can be found in the `LICENSE` file.

