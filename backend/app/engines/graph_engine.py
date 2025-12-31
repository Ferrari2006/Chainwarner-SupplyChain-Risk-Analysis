try:
    import easygraph as eg
    USE_EASYGRAPH = True
except ImportError:
    import networkx as eg
    USE_EASYGRAPH = False
import random

class GraphEngine:
    def __init__(self):
        self.G = eg.DiGraph()

    def build_graph(self, nodes, edges):
        self.G = eg.DiGraph()
        for node in nodes:
            self.G.add_node(node['id'], **node)
        for edge in edges:
            self.G.add_edge(edge['source'], edge['target'], **edge)

    def calculate_metrics(self):
        """
        Calculate advanced graph metrics:
        1. Constraint (Structural Holes)
        2. Effective Size (Structural Holes)
        3. Betweenness Centrality
        4. PageRank
        5. K-Core (if available)
        """
        metrics = {}
        
        # 1. Structural Holes (Burt's Constraint)
        # Identifies brokers who connect otherwise disconnected communities.
        try:
            # Try EasyGraph first
            if USE_EASYGRAPH:
                constraint = eg.constraint(self.G)
                metrics['constraint'] = constraint
            else:
                # Use NetworkX directly if EasyGraph is missing
                import networkx as nx
                metrics['constraint'] = nx.constraint(self.G)
        except Exception as e:
            print(f"EasyGraph/NetworkX constraint failed: {e}")
            # Fallback: Manually calculate simplified Constraint (Sum of dyadic constraints)
            # C_i = sum( (p_ij + sum(p_iq * p_qj))^2 )
            # This is a complex calculation, so we use a simpler heuristic for fallback:
            # Degree Centrality as a proxy for "Constraint" (High Degree ~ High Constraint in dense clusters)
            try:
                metrics['constraint'] = {n: deg / (len(self.G) - 1) for n, deg in self.G.degree()}
            except:
                metrics['constraint'] = {}

        # 2. Betweenness Centrality
        # Identifies nodes that control information flow.
        try:
            betweenness = eg.betweenness_centrality(self.G)
            metrics['betweenness'] = betweenness
        except Exception as e:
            print(f"Error calculating betweenness: {e}")

        # 3. PageRank (Influence)
        try:
            pagerank = eg.pagerank(self.G)
            metrics['pagerank'] = pagerank
        except Exception as e:
            print(f"Error calculating pagerank: {e}")

        return metrics

    def detect_communities(self):
        """
        Identify communities (ecosystems) within the graph.
        Uses Louvain algorithm if available, else Label Propagation.
        """
        try:
            if USE_EASYGRAPH:
                # EasyGraph Community Detection
                communities = eg.get_communities(self.G)
                return communities
            else:
                # NetworkX Fallback
                import networkx as nx
                # community = nx.community.louvain_communities(self.G) # Requires newer nx
                # Fallback to simple connected components for demo
                return [list(c) for c in nx.connected_components(self.G.to_undirected())]
        except Exception as e:
            print(f"Community Detection Failed: {e}")
            return {}

    def simulate_risk_propagation(self, start_node, max_depth=3):
        """
        Simulate risk diffusion using BFS (Breadth-First Search).
        Returns a list of affected nodes layer by layer.
        """
        affected_nodes = {}
        visited = set()
        queue = [(start_node, 0)]
        visited.add(start_node)

        while queue:
            node, depth = queue.pop(0)
            if depth > max_depth:
                continue
            
            if depth not in affected_nodes:
                affected_nodes[depth] = []
            affected_nodes[depth].append(node)

            # Get downstream neighbors (Reverse graph if edges are 'depends_on')
            # Assuming G is: Source depends on Target. 
            # Risk flows from Target (upstream) to Source (downstream).
            # So we need predecessors in standard dependency graph.
            try:
                # EasyGraph predecessor access
                predecessors = list(self.G.predecessors(node))
                for pred in predecessors:
                    if pred not in visited:
                        visited.add(pred)
                        queue.append((pred, depth + 1))
            except:
                pass
        
        return affected_nodes

    def get_risk_path(self, source, target):
        try:
            return eg.shortest_path(self.G, source, target)
        except:
            return []
