from perception import generate_equipment_state, VLMAnalyzer
from engine import SafetyAudit, EnvironmentalConditions, WorkPermit
import networkx as nx
import matplotlib
matplotlib.use('Agg')   # headless — no display needed in WSL2
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv(os.path.expanduser("~/.openclaw/.env"))

# ── Knowledge Graph ───────────────────────────────────────────────────────────

class SafetyKnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
        )

    def build(self, perception: dict, audit: dict):
        with self.driver.session() as s:
            # Clear previous run
            s.run("MATCH (n) DETACH DELETE n")

            # Scene node
            s.run("CREATE (:Scene {id: 'current', timestamp: datetime()})")

            # Valve nodes
            for v in perception.get('valves_open', []):
                s.run("CREATE (:Valve {id: $id, state: 'OPEN'})", id=v)
                s.run("""MATCH (sc:Scene {id:'current'}), (v:Valve {id:$id})
                         CREATE (sc)-[:HAS_VALVE]->(v)""", id=v)
            for v in perception.get('valves_closed', []):
                s.run("CREATE (:Valve {id: $id, state: 'CLOSED'})", id=v)
                s.run("""MATCH (sc:Scene {id:'current'}), (v:Valve {id:$id})
                         CREATE (sc)-[:HAS_VALVE]->(v)""", id=v)

            # Equipment nodes
            s.run("""CREATE (:Equipment {
                        type: 'pressure_gauge',
                        value_psi: $psi})""",
                  psi=perception.get('pressure_psi', 0))
            s.run("""CREATE (:Equipment {
                        type: 'worker',
                        present: $p})""",
                  p=perception.get('worker_present', False))
            s.run("""CREATE (:Equipment {
                        type: 'tools',
                        present: $p})""",
                  p=perception.get('tools_present', False))

            # Violation nodes — linked to scene
            for i, v in enumerate(audit.get('violations', [])):
                s.run("""CREATE (:Violation {
                            id: $id,
                            severity: $sev,
                            description: $desc})""",
                      id=f"V{i}",
                      sev="CRITICAL" if "CRITICAL" in v else "CASCADE",
                      desc=v)
                s.run("""MATCH (sc:Scene {id:'current'}),
                               (vio:Violation {id:$id})
                         CREATE (sc)-[:TRIGGERS]->(vio)""", id=f"V{i}")

            # Model node — shows provenance
            s.run("""CREATE (:AIModel {
                        name: 'Nemotron-Ultra-253B',
                        vlm:  'Llama-3.2-11B-Vision',
                        role: 'perception+reasoning'})""")
            s.run("""MATCH (m:AIModel), (sc:Scene {id:'current'})
                     CREATE (m)-[:ANALYZED]->(sc)""")

        print("✓ Knowledge graph written to Neo4j AuraDB")

    def query_violations(self) -> list:
        """Actually query the graph — this is what makes it a real KG."""
        with self.driver.session() as s:
            result = s.run("""
                MATCH (sc:Scene)-[:TRIGGERS]->(v:Violation)
                RETURN v.severity AS severity,
                       v.description AS description
                ORDER BY severity
            """)
            return [dict(r) for r in result]

    def close(self):
        self.driver.close()


# ── Main Analysis ─────────────────────────────────────────────────────────────

def run_enhanced_analysis():
    print("\n=== Industrial Safety Twin ===")
    print("Model: Nemotron Ultra 253B (NVIDIA API)\n")

    # Scene parameters
    scene_params = {
        'valve_states':   [1, 0, 0, 1],
        'tools_present':  True,
        'person_present': True,
        'pressure_gauge': 55
    }

    # 1. Generate synthetic image (for visual display only)
    img = generate_equipment_state(**scene_params)
    print("✓ Synthetic scene image generated")

    # 2. Run Nemotron perception analysis via API
    print("→ Sending scene to Nemotron 70B for analysis...")
    analyzer  = VLMAnalyzer()
    perception = analyzer.analyze_scene(img)
    print(f"✓ Perception complete")
    print(f"  Valves open:   {perception['valves_open']}")
    print(f"  Valves closed: {perception['valves_closed']}")
    print(f"  Pressure:      {perception['pressure_psi']} PSI")
    print(f"  Worker:        {perception['worker_present']}")
    print(f"  Tools:         {perception['tools_present']}")
    if perception.get('hazards'):
        print(f"  Hazards flagged: {perception['hazards']}")

    # 3. Run semantic safety audit
    env   = EnvironmentalConditions(ventilation="poor")
    audit_engine = SafetyAudit()
    audit = audit_engine.run(perception, env, permit=None)

    print("\n── Safety Audit Results ──")
    if audit['violations']:
        for v in audit['violations']:
            print(f"  ❌ {v}")
    else:
        print("  ✅ No violations detected")
    for w in audit.get('warnings', []):
        print(f"  ⚠️  {w}")

    # 4. Build and save knowledge graph
    kg = SafetyKnowledgeGraph()
    kg.build(perception, audit)
    
    violations_from_graph = kg.query_violations()
    print(f"\n── Graph Query Results ({len(violations_from_graph)} violations) ──")

    for v in violations_from_graph:
        print(f"  [{v['severity']}] {v['description']}")
   
    kg.close()

    return {
        'perception': perception,
        'audit':      audit,
        'scene_image': img
    }

if __name__ == "__main__":
    run_enhanced_analysis()

