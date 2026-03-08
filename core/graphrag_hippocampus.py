# -*- coding: utf-8 -*-
"""
GraphRAG Hippocampus - Axiom-OS v4.0

Knowledge Graph + Vector DB for cross-disciplinary reasoning.

Key innovation: Store physics knowledge as a graph where:
- Nodes: Physical quantities (Force, Velocity, Pressure, Energy...)
- Edges: Mathematical/physical relationships (proportional_to, gradient_of, conserved_in...)

Capabilities:
1. Topological retrieval: Find related concepts via graph traversal
2. Cross-domain reasoning: Link fluid dynamics to electromagnetism via similar math
3. Analogical reasoning: "This looks like Navier-Stokes, maybe try those techniques"

Reference: GraphRAG (Microsoft Research, 2024)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import re
from sklearn.metrics.pairwise import cosine_similarity

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class PhysicalEntity:
    """Node in the knowledge graph - represents a physical quantity or concept"""
    id: str
    name: str
    entity_type: str  # "quantity", "law", "system", "field", "operator"
    domain: str  # "mechanics", "fluids", "electromagnetism", "thermodynamics", ...
    
    # Attributes
    units: Optional[Dict[str, int]] = None  # {M: 1, L: 1, T: -2} for Force
    formula: Optional[str] = None  # Mathematical expression
    description: str = ""
    
    # Vector embedding for semantic search
    embedding: Optional[np.ndarray] = None
    
    # Metadata
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhysicalRelation:
    """Edge in the knowledge graph - represents relationship between entities"""
    source: str  # Source entity ID
    target: str  # Target entity ID
    relation_type: str  # "proportional_to", "gradient_of", "conserved_in", "analogous_to", ...
    
    # Relation attributes
    weight: float = 1.0  # Strength of relationship
    formula: Optional[str] = None  # Mathematical relationship
    conditions: str = ""  # When does this relationship hold?
    
    # Cross-domain mapping
    is_analogy: bool = False  # Is this a cross-domain analogy?
    analogy_score: float = 0.0  # Confidence in analogy


class PhysicsKnowledgeGraph:
    """
    Knowledge graph for physics concepts and relationships.
    
    Enables:
    - Topological retrieval (graph traversal)
    - Semantic retrieval (vector similarity)
    - Cross-domain analogies (structural matching)
    """
    
    # Standard physical dimensions
    DIMENSIONS = ['M', 'L', 'T', 'Q', 'Theta']  # Mass, Length, Time, Charge, Temperature
    
    # Domain taxonomy
    DOMAINS = {
        'mechanics': {'parent': None, 'color': '#3498db'},
        'fluids': {'parent': 'mechanics', 'color': '#2980b9'},
        'electromagnetism': {'parent': None, 'color': '#e74c3c'},
        'thermodynamics': {'parent': None, 'color': '#f39c12'},
        'quantum': {'parent': None, 'color': '#9b59b6'},
        'relativity': {'parent': 'mechanics', 'color': '#1abc9c'},
    }
    
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for causal relationships
        self.entities: Dict[str, PhysicalEntity] = {}
        self.relations: List[PhysicalRelation] = []
        
        # Embedding cache
        self._embeddings: Dict[str, np.ndarray] = {}
        
        # Initialize with core physics concepts
        self._initialize_core_knowledge()
    
    def _initialize_core_knowledge(self):
        """Initialize graph with fundamental physics concepts"""
        
        # Core quantities
        core_entities = [
            # Mechanics
            PhysicalEntity("force", "Force", "quantity", "mechanics", 
                          units={'M': 1, 'L': 1, 'T': -2}, 
                          formula="F = m*a", 
                          description="Newton's second law"),
            PhysicalEntity("mass", "Mass", "quantity", "mechanics", 
                          units={'M': 1}),
            PhysicalEntity("velocity", "Velocity", "quantity", "mechanics",
                          units={'L': 1, 'T': -1}),
            PhysicalEntity("acceleration", "Acceleration", "quantity", "mechanics",
                          units={'L': 1, 'T': -2}),
            PhysicalEntity("energy", "Energy", "quantity", "mechanics",
                          units={'M': 1, 'L': 2, 'T': -2}),
            PhysicalEntity("momentum", "Momentum", "quantity", "mechanics",
                          units={'M': 1, 'L': 1, 'T': -1}),
            
            # Fluids
            PhysicalEntity("pressure", "Pressure", "quantity", "fluids",
                          units={'M': 1, 'L': -1, 'T': -2}),
            PhysicalEntity("viscosity", "Viscosity", "quantity", "fluids",
                          units={'M': 1, 'L': -1, 'T': -1}),
            PhysicalEntity("density", "Density", "quantity", "fluids",
                          units={'M': 1, 'L': -3}),
            PhysicalEntity("velocity_field", "Velocity Field", "field", "fluids",
                          units={'L': 1, 'T': -1}),
            PhysicalEntity("vorticity", "Vorticity", "quantity", "fluids",
                          units={'T': -1}),
            
            # Electromagnetism
            PhysicalEntity("electric_field", "Electric Field", "field", "electromagnetism",
                          units={'M': 1, 'L': 1, 'T': -3, 'Q': -1}),
            PhysicalEntity("magnetic_field", "Magnetic Field", "field", "electromagnetism",
                          units={'M': 1, 'T': -2, 'Q': -1}),
            PhysicalEntity("charge", "Charge", "quantity", "electromagnetism",
                          units={'Q': 1}),
            PhysicalEntity("current", "Current", "quantity", "electromagnetism",
                          units={'Q': 1, 'T': -1}),
            
            # Laws/Equations
            PhysicalEntity("navier_stokes", "Navier-Stokes Equations", "law", "fluids",
                          formula="du/dt + u·∇u = -∇p/ρ + ν∇²u",
                          description="Momentum conservation in fluid flow"),
            PhysicalEntity("maxwell", "Maxwell's Equations", "law", "electromagnetism",
                          formula="∇·E = ρ/ε₀, ∇×E = -∂B/∂t, ...",
                          description="Classical electromagnetism"),
            PhysicalEntity("heat_equation", "Heat Equation", "law", "thermodynamics",
                          formula="∂T/∂t = α∇²T",
                          description="Thermal diffusion"),
            PhysicalEntity("wave_equation", "Wave Equation", "law", "mechanics",
                          formula="∂²u/∂t² = c²∇²u",
                          description="Wave propagation"),
            
            # Operators
            PhysicalEntity("gradient", "Gradient", "operator", "mechanics",
                          formula="∇f"),
            PhysicalEntity("divergence", "Divergence", "operator", "mechanics",
                          formula="∇·F"),
            PhysicalEntity("curl", "Curl", "operator", "mechanics",
                          formula="∇×F"),
            PhysicalEntity("laplacian", "Laplacian", "operator", "mechanics",
                          formula="∇²f"),
        ]
        
        for entity in core_entities:
            self.add_entity(entity)
        
        # Core relationships
        core_relations = [
            # Force relationships
            PhysicalRelation("force", "mass", "depends_on", weight=1.0),
            PhysicalRelation("force", "acceleration", "proportional_to", weight=1.0, 
                           formula="F = m*a"),
            
            # Energy relationships
            PhysicalRelation("energy", "force", "related_to", weight=0.8),
            PhysicalRelation("energy", "velocity", "proportional_to_squared",
                           formula="E_k = ½mv²"),
            
            # Fluid dynamics
            PhysicalRelation("pressure", "force", "gradient_of", weight=0.9,
                           formula="F = -∇p"),
            PhysicalRelation("viscosity", "stress", "proportional_to", weight=0.8,
                           formula="τ = μ∇u"),
            PhysicalRelation("navier_stokes", "velocity_field", "governs", weight=1.0),
            PhysicalRelation("navier_stokes", "pressure", "involves", weight=0.9),
            PhysicalRelation("navier_stokes", "viscosity", "involves", weight=0.9),
            PhysicalRelation("vorticity", "velocity_field", "curl_of", weight=1.0,
                           formula="ω = ∇×u"),
            
            # Operator relationships
            PhysicalRelation("divergence", "gradient", "composition_of", weight=0.7),
            PhysicalRelation("curl", "gradient", "orthogonal_to", weight=0.7),
            PhysicalRelation("laplacian", "gradient", "divergence_of_gradient", weight=1.0,
                           formula="∇²f = ∇·(∇f)"),
            
            # Cross-domain analogies (key for GraphRAG!)
            PhysicalRelation("navier_stokes", "maxwell", "analogous_to", weight=0.6,
                           is_analogy=True, analogy_score=0.7,
                           conditions="Both involve field evolution with source terms"),
            PhysicalRelation("velocity_field", "magnetic_field", "analogous_to", weight=0.5,
                           is_analogy=True, analogy_score=0.6,
                           conditions="Both are vector fields with divergence/curl structure"),
            PhysicalRelation("vorticity", "current", "analogous_to", weight=0.5,
                           is_analogy=True, analogy_score=0.6,
                           conditions="Both related to curl of underlying field"),
            PhysicalRelation("heat_equation", "wave_equation", "analogous_to", weight=0.4,
                           is_analogy=True, analogy_score=0.5,
                           conditions="Both are PDEs with Laplacian operator"),
        ]
        
        for relation in core_relations:
            self.add_relation(relation)
    
    def add_entity(self, entity: PhysicalEntity):
        """Add an entity to the graph"""
        self.entities[entity.id] = entity
        self.graph.add_node(entity.id, **{
            'name': entity.name,
            'type': entity.entity_type,
            'domain': entity.domain,
            'formula': entity.formula,
        })
    
    def add_relation(self, relation: PhysicalRelation):
        """Add a relationship to the graph"""
        self.relations.append(relation)
        self.graph.add_edge(
            relation.source, relation.target,
            relation_type=relation.relation_type,
            weight=relation.weight,
            is_analogy=relation.is_analogy,
        )
    
    def get_neighbors(self, entity_id: str, relation_type: Optional[str] = None) -> List[str]:
        """Get neighboring entities"""
        if entity_id not in self.graph:
            return []
        
        neighbors = []
        for neighbor in self.graph.successors(entity_id):
            edge_data = self.graph[entity_id][neighbor]
            if relation_type is None or edge_data.get('relation_type') == relation_type:
                neighbors.append(neighbor)
        return neighbors
    
    def traverse(self, start_id: str, depth: int = 2, 
                 relation_filter: Optional[List[str]] = None) -> Dict[str, int]:
        """
        BFS traversal from starting entity.
        
        Returns:
            Dict mapping entity_id to distance from start
        """
        if start_id not in self.graph:
            return {}
        
        visited = {start_id: 0}
        queue = [(start_id, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            if dist >= depth:
                continue
            
            for neighbor in self.graph.successors(current):
                if neighbor in visited:
                    continue
                
                edge_data = self.graph[current][neighbor]
                if relation_filter and edge_data.get('relation_type') not in relation_filter:
                    continue
                
                visited[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
        
        return visited
    
    def find_analogies(self, entity_id: str, min_score: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find cross-domain analogies for an entity.
        
        Returns:
            List of (analogous_entity_id, score) tuples
        """
        analogies = []
        for relation in self.relations:
            if relation.is_analogy and relation.analogy_score >= min_score:
                if relation.source == entity_id:
                    analogies.append((relation.target, relation.analogy_score))
                elif relation.target == entity_id:
                    analogies.append((relation.source, relation.analogy_score))
        return sorted(analogies, key=lambda x: -x[1])
    
    def get_domain_entities(self, domain: str, entity_type: Optional[str] = None) -> List[PhysicalEntity]:
        """Get all entities in a domain"""
        results = []
        for entity in self.entities.values():
            if entity.domain == domain:
                if entity_type is None or entity.entity_type == entity_type:
                    results.append(entity)
        return results
    
    def check_dimensional_consistency(self, formula_units: Dict[str, int]) -> List[PhysicalEntity]:
        """
        Find entities with matching dimensional units.
        Useful for finding physically meaningful combinations.
        """
        matches = []
        for entity in self.entities.values():
            if entity.units and self._units_match(entity.units, formula_units):
                matches.append(entity)
        return matches
    
    def _units_match(self, units1: Dict[str, int], units2: Dict[str, int]) -> bool:
        """Check if two unit dictionaries match"""
        for dim in self.DIMENSIONS:
            if units1.get(dim, 0) != units2.get(dim, 0):
                return False
        return True


class GraphRAGRetriever:
    """
    GraphRAG retrieval engine.
    Combines topological (graph) and semantic (vector) retrieval.
    """
    
    def __init__(self, knowledge_graph: PhysicsKnowledgeGraph):
        self.kg = knowledge_graph
        
        # Simple keyword-based embedding (can be replaced with real embeddings)
        self.vocabulary = self._build_vocabulary()
        self.entity_embeddings = self._compute_entity_embeddings()
    
    def _build_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary from entity names and descriptions"""
        vocab = {}
        for entity in self.kg.entities.values():
            words = self._tokenize(entity.name + " " + entity.description)
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    def _compute_entity_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute bag-of-words embeddings for entities"""
        embeddings = {}
        for entity_id, entity in self.kg.entities.items():
            text = entity.name + " " + entity.description + " " + str(entity.formula or "")
            words = self._tokenize(text)
            
            vec = np.zeros(len(self.vocabulary))
            for word in words:
                if word in self.vocabulary:
                    vec[self.vocabulary[word]] += 1
            
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            
            embeddings[entity_id] = vec
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a text query"""
        words = self._tokenize(query)
        vec = np.zeros(len(self.vocabulary))
        for word in words:
            if word in self.vocabulary:
                vec[self.vocabulary[word]] += 1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_graph: bool = True,
        use_semantic: bool = True,
        exploration_depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        GraphRAG retrieval combining topological and semantic search.
        
        Args:
            query: Text query (e.g., "pressure gradient in fluid flow")
            top_k: Number of results
            use_graph: Whether to use graph traversal
            use_semantic: Whether to use semantic similarity
            exploration_depth: Graph traversal depth
        
        Returns:
            List of results with scores and paths
        """
        results = []
        query_vec = self.embed_query(query)
        
        # Find initial matches via semantic similarity
        semantic_scores = {}
        for entity_id, emb in self.entity_embeddings.items():
            sim = float(np.dot(query_vec, emb))
            if sim > 0.1:  # Threshold
                semantic_scores[entity_id] = sim
        
        # Sort by semantic score
        sorted_entities = sorted(semantic_scores.items(), key=lambda x: -x[1])
        
        # For top matches, explore graph neighborhood
        for entity_id, sem_score in sorted_entities[:top_k]:
            entity = self.kg.entities[entity_id]
            
            result = {
                'entity': entity,
                'semantic_score': sem_score,
                'graph_neighbors': [],
                'analogies': [],
            }
            
            if use_graph:
                # Traverse graph
                neighbors = self.kg.traverse(entity_id, depth=exploration_depth)
                for neighbor_id, distance in neighbors.items():
                    if neighbor_id != entity_id:
                        neighbor = self.kg.entities.get(neighbor_id)
                        if neighbor:
                            result['graph_neighbors'].append({
                                'entity': neighbor,
                                'distance': distance,
                            })
                
                # Find analogies (cross-domain)
                analogies = self.kg.find_analogies(entity_id, min_score=0.5)
                for analog_id, score in analogies[:3]:  # Top 3 analogies
                    analog = self.kg.entities.get(analog_id)
                    if analog:
                        result['analogies'].append({
                            'entity': analog,
                            'analogy_score': score,
                        })
            
            results.append(result)
        
        return results
    
    def suggest_cross_domain_transfer(self, source_domain: str, target_domain: str,
                                     concept: str) -> List[Dict[str, Any]]:
        """
        Suggest techniques from source_domain that might apply to target_domain.
        
        Example: suggest_cross_domain_transfer("electromagnetism", "fluids", "field evolution")
        Might suggest: "Try divergence-free formulations like in Maxwell's equations"
        """
        suggestions = []
        
        # Get concepts in source domain
        source_concepts = self.kg.get_domain_entities(source_domain)
        
        # Find analogies to target domain
        for concept_entity in source_concepts:
            analogies = self.kg.find_analogies(concept_entity.id, min_score=0.4)
            for analog_id, score in analogies:
                analog = self.kg.entities.get(analog_id)
                if analog and analog.domain == target_domain:
                    suggestions.append({
                        'source_concept': concept_entity,
                        'target_analog': analog,
                        'similarity_score': score,
                        'suggestion': f"{concept_entity.name} ({source_domain}) → {analog.name} ({target_domain})",
                    })
        
        return sorted(suggestions, key=lambda x: -x['similarity_score'])


class GraphRAGHippocampus:
    """
    GraphRAG-enhanced Hippocampus for Axiom-OS v4.0.
    
    Replaces simple dict storage with knowledge graph for:
    - Topological retrieval (graph traversal)
    - Cross-domain reasoning (analogies)
    - Structured knowledge (entities, relations, units)
    """
    
    def __init__(self):
        self.kg = PhysicsKnowledgeGraph()
        self.retriever = GraphRAGRetriever(self.kg)
        
        # Also keep original memory for compatibility
        self._memory: List[Tuple[np.ndarray, Any, float]] = []
        self.capacity = 5000
    
    def store_crystallized_law(
        self,
        formula: str,
        domain: str,
        name: Optional[str] = None,
        units: Optional[Dict[str, int]] = None,
        related_concepts: Optional[List[str]] = None,
    ) -> str:
        """
        Store a discovered physical law in the knowledge graph.
        
        Args:
            formula: Mathematical formula string
            domain: Physics domain
            name: Human-readable name
            units: Dimensional units
            related_concepts: List of related entity IDs
        
        Returns:
            Entity ID
        """
        entity_id = name.lower().replace(" ", "_") if name else f"law_{len(self.kg.entities)}"
        
        entity = PhysicalEntity(
            id=entity_id,
            name=name or f"Law {entity_id}",
            entity_type="law",
            domain=domain,
            formula=formula,
            units=units,
        )
        
        self.kg.add_entity(entity)
        
        # Add relations to related concepts
        if related_concepts:
            for related_id in related_concepts:
                if related_id in self.kg.entities:
                    relation = PhysicalRelation(
                        source=entity_id,
                        target=related_id,
                        relation_type="related_to",
                        weight=0.8,
                    )
                    self.kg.add_relation(relation)
        
        # Update retriever embeddings
        self.retriever.entity_embeddings = self.retriever._compute_entity_embeddings()
        
        return entity_id
    
    def retrieve_by_query(
        self,
        question: str,
        top_k: int = 5,
        include_analogies: bool = True,
    ) -> str:
        """
        GraphRAG retrieval with cross-domain reasoning.
        
        Args:
            question: Natural language query
            top_k: Number of results
            include_analogies: Whether to include cross-domain analogies
        
        Returns:
            Formatted response string
        """
        results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            use_graph=True,
            use_semantic=True,
        )
        
        if not results:
            return "No relevant knowledge found."
        
        lines = ["Known laws and concepts (GraphRAG):", ""]
        
        for i, result in enumerate(results, 1):
            entity = result['entity']
            lines.append(f"{i}. {entity.name} ({entity.domain})")
            formula_str = entity.formula or 'N/A'
            # Replace unicode characters for console compatibility
            formula_str = formula_str.replace('∇', 'grad').replace('∂', 'd').replace('²', '^2')
            lines.append(f"   Formula: {formula_str}")
            lines.append(f"   Semantic relevance: {result['semantic_score']:.3f}")
            
            if result['graph_neighbors']:
                lines.append("   Related concepts:")
                for neighbor in result['graph_neighbors'][:3]:
                    lines.append(f"     - {neighbor['entity'].name} ({neighbor['entity'].domain})")
            
            if include_analogies and result['analogies']:
                lines.append("   Cross-domain analogies:")
                for analog in result['analogies'][:2]:
                    lines.append(f"     → {analog['entity'].name} ({analog['entity'].domain}) "
                               f"[score: {analog['analogy_score']:.2f}]")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def suggest_transfer_learning(self, current_domain: str, problem_description: str) -> str:
        """
        Suggest techniques from other domains that might help.
        
        Example: "I'm working on turbulence (fluids), what techniques from other domains might help?"
        """
        suggestions = []
        
        # Check all other domains
        for other_domain in self.kg.DOMAINS.keys():
            if other_domain == current_domain:
                continue
            
            domain_suggestions = self.retriever.suggest_cross_domain_transfer(
                other_domain, current_domain, problem_description
            )
            
            if domain_suggestions:
                suggestions.extend(domain_suggestions)
        
        if not suggestions:
            return f"No clear cross-domain analogies found for {current_domain}."
        
        lines = [f"Cross-domain suggestions for {current_domain}:", ""]
        
        for sugg in suggestions[:5]:  # Top 5
            lines.append(f"From {sugg['source_concept'].domain}:")
            lines.append(f"  '{sugg['source_concept'].name}' → '{sugg['target_analog'].name}'")
            lines.append(f"  Similarity: {sugg['similarity_score']:.2f}")
            lines.append(f"  Suggestion: {sugg['suggestion']}")
            lines.append("")
        
        lines.append("Consider adapting techniques from these analogous domains!")
        
        return "\n".join(lines)
    
    # Compatibility with original Hippocampus interface
    def store(self, key: np.ndarray, value: np.ndarray, label: Any = None, confidence: float = 1.0):
        """Store in original memory (backward compatibility)"""
        if len(self._memory) >= self.capacity:
            self._memory.pop(0)
        self._memory.append((np.asarray(key), value, confidence))
    
    def retrieve(self, query: np.ndarray, top_k: int = 5):
        """Retrieve from original memory (backward compatibility)"""
        if not self._memory:
            return [], []
        q = np.asarray(query).ravel()
        scores = [float(-np.linalg.norm(np.asarray(emb).ravel() - q)) 
                 for k, val, emb in self._memory if emb is not None]
        idx = np.argsort(scores)[::-1][:top_k]
        return [scores[i] for i in idx], [self._memory[i][1] for i in idx]


def test_graphrag_hippocampus():
    """Test GraphRAG Hippocampus"""
    print("=" * 70)
    print("GraphRAG Hippocampus Test - Axiom-OS v4.0")
    print("=" * 70)
    
    # Create GraphRAG Hippocampus
    hippo = GraphRAGHippocampus()
    
    print(f"\n[1] Knowledge Graph initialized")
    print(f"    Entities: {len(hippo.kg.entities)}")
    print(f"    Relations: {len(hippo.kg.relations)}")
    
    # Test semantic retrieval
    print("\n[2] Testing semantic retrieval...")
    query = "pressure gradient fluid flow"
    results = hippo.retriever.retrieve(query, top_k=3)
    print(f"    Query: '{query}'")
    for i, r in enumerate(results, 1):
        print(f"    {i}. {r['entity'].name} (score: {r['semantic_score']:.3f})")
    
    # Test graph traversal
    print("\n[3] Testing graph traversal...")
    neighbors = hippo.kg.traverse("navier_stokes", depth=2)
    print(f"    From 'Navier-Stokes Equations' (depth 2):")
    for entity_id, dist in list(neighbors.items())[:5]:
        entity = hippo.kg.entities.get(entity_id)
        if entity:
            print(f"      {'  ' * dist}{entity.name}")
    
    # Test analogies
    print("\n[4] Testing cross-domain analogies...")
    analogies = hippo.kg.find_analogies("navier_stokes", min_score=0.5)
    print(f"    Analogies to 'Navier-Stokes Equations':")
    for analog_id, score in analogies:
        analog = hippo.kg.entities.get(analog_id)
        if analog:
            print(f"      → {analog.name} ({analog.domain}) [score: {score:.2f}]")
    
    # Test full GraphRAG retrieval
    print("\n[5] Testing full GraphRAG retrieval...")
    response = hippo.retrieve_by_query("viscous stress in turbulence", top_k=3)
    print(response)
    
    # Test cross-domain suggestions
    print("\n[6] Testing cross-domain transfer suggestions...")
    suggestion = hippo.suggest_transfer_learning("fluids", "field evolution")
    print(suggestion)
    
    # Test storing new law
    print("\n[7] Testing store_crystallized_law...")
    law_id = hippo.store_crystallized_law(
        formula="τ = -2(CsΔ)²|S|S",
        domain="fluids",
        name="Smagorinsky Model",
        units={'M': 1, 'L': -1, 'T': -2},
        related_concepts=["viscosity", "velocity_field", "navier_stokes"],
    )
    print(f"    Stored law: {law_id}")
    print(f"    Total entities: {len(hippo.kg.entities)}")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] GraphRAG Hippocampus ready!")
    print("=" * 70)
    print("\nKey Features:")
    print("  - Knowledge graph with 20+ core physics entities")
    print("  - Topological retrieval (graph traversal)")
    print("  - Semantic retrieval (vector similarity)")
    print("  - Cross-domain analogies (fluids <-> electromagnetism)")
    print("  - Structured storage (entities, relations, units)")
    print("  - Dimensional consistency checking")


if __name__ == "__main__":
    test_graphrag_hippocampus()
