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
        import os
        lite_mode = os.getenv("LITE_MODE", "false").lower() == "true"  # 默认关闭 lite_mode 以启用完整计算
        node_count = self.G.number_of_nodes()
        nx_g = None
        try:
            nodes_iter = list(self.G.nodes)
        except TypeError:
            nodes_iter = list(self.G.nodes())
        try:
            edges_iter = list(self.G.edges)
        except TypeError:
            try:
                edges_iter = list(self.G.edges())
            except Exception:
                edges_iter = []
        except Exception:
            edges_iter = []
        
        # 1. Structural Holes (Burt's Constraint)
        # Identifies brokers who connect otherwise disconnected communities.
        # OPTIMIZATION: Skip constraint only on very large graphs (it's O(V^3) or O(VE))
        if node_count < 500:  # 提高阈值以支持更大图
            try:
                # Compute constraint on an undirected copy (NetworkX) to avoid
                # directed/disconnected artifacts and keep results consistent.
                try:
                    import networkx as nx
                    nx_g = nx.Graph()
                    nx_g.add_nodes_from(nodes_iter)
                    # Add edges as undirected (only unique pairs)
                    for u, v in edges_iter:
                        try:
                            nx_g.add_edge(u, v)
                        except:
                            pass

                    # Constraint is defined as sum( (p_ij + sum(p_iq * p_qj))^2 )
                    # It requires connected components or handled per ego-net.
                    # nx.constraint handles this but returns NaN for isolates.
                    raw_constraint = nx.constraint(nx_g)

                    import math

                    metrics['constraint'] = {}
                    for n in nx_g.nodes():
                        val = raw_constraint.get(n, float("nan"))
                        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                            degree = nx_g.degree(n)
                            val = 1.0 if degree == 0 else 0.0
                        metrics["constraint"][n] = float(val)
                except Exception as e:
                    print(f"Constraint (NetworkX) calculation failed: {e}")
                    metrics["constraint"] = {}
                    for n in nodes_iter:
                        degree = self.G.degree(n)
                        metrics["constraint"][n] = 1.0 / (degree + 1) if degree > 0 else 1.0
            except Exception as e:
                print(f"EasyGraph/NetworkX constraint failed: {e}")
                metrics["constraint"] = {n: 1.0 for n in nodes_iter}
        else:
            metrics["constraint"] = {}
            for n in nodes_iter:
                degree = self.G.degree(n)
                metrics["constraint"][n] = 1.0 / (degree + 1) if degree > 0 else 1.0

        if node_count < 500 and nx_g is not None:
            try:
                import networkx as nx
                raw_effective_size = nx.effective_size(nx_g)
                metrics["effective_size"] = {}
                for n in nx_g.nodes():
                    val = raw_effective_size.get(n, 0.0)
                    if val is None:
                        val = float(nx_g.degree(n))
                    metrics["effective_size"][n] = float(val)
            except Exception as e:
                print(f"Effective Size calculation failed: {e}")
                metrics["effective_size"] = {n: float(self.G.degree(n)) for n in nodes_iter}
        else:
            metrics["effective_size"] = {n: float(self.G.degree(n)) for n in nodes_iter}

        if node_count < 200:
            try:
                betweenness = eg.betweenness_centrality(self.G)
                metrics["betweenness"] = betweenness
            except Exception as e:
                print(f"Error calculating betweenness: {e}")
        else:
            metrics["betweenness"] = {n: 0.0 for n in nodes_iter}

        # 3. PageRank (Influence)
        # PageRank is relatively fast (iterative), usually safe to keep
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
