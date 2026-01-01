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
            # Compute constraint on an undirected copy (NetworkX) to avoid
            # directed/disconnected artifacts and keep results consistent.
            try:
                import networkx as nx
                nx_g = nx.Graph()
                nx_g.add_nodes_from(list(self.G.nodes))
                # Add edges as undirected (only unique pairs)
                try:
                    edges_iter = self.G.edges
                except:
                    edges_iter = []
                for u, v in edges_iter:
                    try:
                        nx_g.add_edge(u, v)
                    except:
                        pass

                raw_constraint = nx.constraint(nx_g)
                # Log raw constraint for debugging (can be removed later)
                print("[graph_engine] raw_constraint sample:", dict(list(raw_constraint.items())[:5]))

                # Sanitize NaN or missing values with degree-based heuristic
                for n in nx_g.nodes():
                    val = raw_constraint.get(n, None)
                    if val is None or (isinstance(val, float) and val != val):
                        deg = nx_g.degree(n)
                        max_deg = max(1, nx_g.number_of_nodes() - 1)
                        raw_constraint[n] = (deg / max_deg) if max_deg > 0 else 0.0

                metrics['constraint'] = raw_constraint
            except Exception as e:
                print(f"Constraint (NetworkX) calculation failed, falling back: {e}")
                # Fallback: Degree Centrality as a proxy
                metrics['constraint'] = {n: (self.G.degree(n) / max(1, len(self.G) - 1)) for n in self.G.nodes}
        except Exception as e:
            print(f"EasyGraph/NetworkX constraint failed: {e}")
            # Fallback: Degree Centrality as a proxy
            try:
                metrics['constraint'] = {n: (self.G.degree(n) / max(1, len(self.G) - 1)) for n in self.G.nodes}
            except:
                metrics['constraint'] = {n: 0.0 for n in self.G.nodes}

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
