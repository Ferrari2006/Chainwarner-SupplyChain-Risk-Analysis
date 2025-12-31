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
            if USE_EASYGRAPH:
                # EasyGraph implementation
                # Note: Some versions might have different API signatures
                constraint = eg.constraint(self.G)
                metrics['constraint'] = constraint
            else:
                # NetworkX fallback
                import networkx as nx
                metrics['constraint'] = nx.constraint(self.G)
        except Exception as e:
            print(f"Error calculating constraint: {e}")

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

    def get_risk_path(self, source, target):
        try:
            return eg.shortest_path(self.G, source, target)
        except:
            return []
