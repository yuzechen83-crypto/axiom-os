"""
Axiom-OS v4.0 Import Verification Test
"""

print("="*60)
print("Axiom-OS v4.0 Import Verification")
print("="*60)

# Test imports for all v4.0 modules
results = {}
errors = []

# 1. Control: Diffusion Policy
print("\n1. Testing Diffusion Policy...")
try:
    from axiom_os.orchestrator.diffusion_policy_simple import (
        DiffusionPolicy, 
        DiffusionPolicyController,
        DiffusionPolicyConfig
    )
    results["Diffusion Policy"] = "PASS"
    print("   DiffusionPolicy: OK")
except Exception as e:
    results["Diffusion Policy"] = "FAIL"
    errors.append(f"DiffusionPolicy: {e}")
    print(f"   Error: {e}")

# 2. Discovery: FunSearch
print("\n2. Testing FunSearch Discovery...")
try:
    from axiom_os.discovery.funsearch_discovery import (
        FunSearchDiscovery,
        FunSearchCandidate,
        LLMFormulaGenerator,
        CodeEvaluator,
        UPIChecker,
        DiscoveryStatus
    )
    results["FunSearch"] = "PASS"
    print("   FunSearchDiscovery: OK")
except Exception as e:
    results["FunSearch"] = "FAIL"
    errors.append(f"FunSearch: {e}")
    print(f"   Error: {e}")

# 3. Perception: Clifford Neural Operator
print("\n3. Testing Clifford Neural Operator...")
try:
    from axiom_os.layers.clifford_neural_operator import (
        CliffordNeuralOperator3D,
        CliffordSpectralConv3D,
        CliffordAlgebra3D
    )
    results["CNO (Clifford)"] = "PASS"
    print("   CliffordNeuralOperator3D: OK")
except Exception as e:
    results["CNO (Clifford)"] = "FAIL"
    errors.append(f"CNO: {e}")
    print(f"   Error: {e}")

# 4. Memory: GraphRAG Hippocampus
print("\n4. Testing GraphRAG Hippocampus...")
try:
    from axiom_os.core.graphrag_hippocampus import (
        GraphRAGHippocampus,
        PhysicsKnowledgeGraph,
        PhysicalEntity,
        PhysicalRelation,
        GraphRAGRetriever
    )
    results["GraphRAG"] = "PASS"
    print("   GraphRAGHippocampus: OK")
except Exception as e:
    results["GraphRAG"] = "FAIL"
    errors.append(f"GraphRAG: {e}")
    print(f"   Error: {e}")

# 5. Simulation: Differentiable Physics
print("\n5. Testing Differentiable Physics...")
try:
    from axiom_os.core.differentiable_physics import (
        DifferentiableRigidBodyDynamics,
        PhysicsConfig,
        RigidBodyState,
        HAS_WARP
    )
    results["Diff. Physics"] = "PASS"
    print(f"   DifferentiableRigidBodyDynamics: OK (Warp={HAS_WARP})")
except Exception as e:
    results["Diff. Physics"] = "FAIL"
    errors.append(f"Diff.Physics: {e}")
    print(f"   Error: {e}")

# 6. Core RCLN v3
print("\n6. Testing RCLN v3 Core...")
try:
    from axiom_os.core import (
        RCLN3,
        GENERICLayer,
        DifferentiableRigidBodyDynamics
    )
    results["RCLN v3 Core"] = "PASS"
    print("   RCLN3: OK")
except Exception as e:
    results["RCLN v3 Core"] = "FAIL"
    errors.append(f"RCLN v3: {e}")
    print(f"   Error: {e}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
passed = sum(1 for r in results.values() if r == "PASS")
total = len(results)
for module, status in results.items():
    symbol = "OK" if status == "PASS" else "X"
    print(f"{module:20s}: [{symbol}] {status}")
print("-"*60)
print(f"Result: {passed}/{total} modules passed")

if errors:
    print("\nErrors:")
    for e in errors:
        print(f"  - {e}")

if passed == total:
    print("\n*** Axiom-OS v4.0 upgrade verification: ALL PASSED ***")
else:
    print(f"\n*** Axiom-OS v4.0 upgrade verification: {total-passed} failures ***")
