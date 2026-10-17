## International Journal of Combinatorial Optimization Problems and Informatics, 18(1), 2026, 17-25. ISSN: 2007-1558.

Doi

## A multi-demand adaptation of Dijkstra’s algorithm for Residential Cable Routing Optimisation

José Enrique Rojas Macedo1, René A. García-Hernández1, Yulia Ledeneva1

1 Universidad Autónoma del Estado de México, Instituto Literario 100, Toluca de Lerdo, México Email address(es): jrojasm014@alumno.uaemex.mx, reagarciah@uaemex.mx, ynledeneva@uaemex.mx [URL 🔗](mailto:jrojasm014@alumno.uaemex.mx)

Corresponding author: jrojasm014@alumno.uaemex.mx, ynledeneva@uaemex.mx [URL 🔗](mailto:jrojasm014@alumno.uaemex.mx)

Abstract. Residential cable routing requires connecting a distribution panel to every electrical demand point through a fixed, pre-existing conduit network at minimum cable length. Dijkstra’s algorithm, in its classical form, cannot solve this problem directly because it is built around a single origin and a single destination, whereas cable routing must reach an entire group of demand points at once. This article identifies precisely why the direct application fails and adapts Dijkstra’s algorithm by redefining the search state to track which demand points have already been connected, and by treating an already connected conduit segment as reusable rather than re-counted. The search is exact for the total length of the walk it explores, and the cost of the tree it reports is at most twice the optimum. The adapted algorithm is evaluated on seven demand configurations defined over two residential floor plans, reused together with their human reference solutions. As results the adaptation returns a shorter cable layout than the human reference in every configuration it solves, by

Received: Dec 26, 2016

Accepted: Dec 11, 2017

Article information

between

3.3%

and

17.5%.

Keywords: Residential cable routing; Dijkstra algorithm; multi-demand shortest path; optimisation; graph search.

## 1 Introduction

The design and installation of electrical wiring in residential buildings involves considerable planning effort. Conduits (plastic or metallic tubes through which electric cables are routed) are typically embedded in walls, floors, and ceilings during construction, before any wiring takes place. Once the conduit infrastructure is fixed, an electrician must decide how to pull cables through it to energise every electrical point (outlet, switch, luminaires, etc.) from the main distribution panel. A poor cabling plan leads to excess material consumption, longer installation time, and higher labour costs.

The problem of finding a minimum-length cable layout over a fixed conduit network is an optimisation problem. The conduit network can be modelled as a weighted undirected graph G = (V, E), where each node in V corresponds to a conduit junction or an electrical point, each edge in E corresponds to a conduit segment, and edge weights represent physical lengths. The optimisation task is to select a minimum weight connected subgraph of G that includes the distribution panel and every demand node. This basically means the Residential Cable Routing Problem (RCRP), and it is the problem this article addresses.

A natural question is whether an established shortest-path algorithm can simply be applied to RCRP as it is. Dijkstra’s algorithm is the most suitable for this scenario as it is well understood, always returns a solution and it is already used in many network-design and routing applications, but it cannot be applied directly to this problem, because it is built around a single objective: it starts at one origin and finds the shortest route to one destination. Residential cable routing has, by contrast, many objectives at once: the distribution panel must be connected to every demand node, in a set that may contain two or more electrical points, not to any of them. Treating this as a routing problem, requires the search itself to change, not only its stopping condition, but what counts as a “position” during the search and how the cost of a partial solution is measured, since a conduit segment that has been already wired can carry the connections for other demand points without being paid for twice.

This article makes an adaptation of Dijkstra’s algorithm, that redefines the search state to track connected demand nodes and treats already connected segments as a reusable resource rather than a repeated cost, and an evaluation of the proposed


adaptation on residential floor plans with their human references’ solutions, together with an analysis of its optimality guarantees and its time and space complexity.

## 2 Related work

Automating the routing of cables and pipes inside buildings has received sustained attention in the engineering informatics literature in recent years, driven largely by the adoption of Building Information Modelling (BIM) and by the recognised cost of manual, experience-based routing. Deng et al. (2025) systematically reviewed path-optimisation algorithms applied to mechanical, electrical, and plumbing design inside buildings, and reported that graph search methods, like Dijkstra’s algorithm and A*, remain the dominant approach for automated routing.

Within this space, several recent studies address problems structurally close to RCRP. Zhou et al. (2023) proposed IEDW, a BIM-based algorithm for indoor electric distribution wiring that models the building as a graph and solves the wiring problem with a capacity-limited multiple travelling-salesman formulation, reporting an 8.8% average reduction in wiring cost relative to manual design on real building cases; unlike RCRP, their formulation allows conductor capacity constraints but treats each circuit largely as a separate routing sub-problem rather than a single tree connecting all demand points from one panel. Choi et al. (2022) modified the A* algorithm for building MEP path design by revising its node selection and post processing steps, and validated the modification on seven real building structures, showing that a general shortest path algorithm needs problem specific revision, not merely parameter tuning, before it produces usable routing output.

Outside building scale MEP design, the closely related problem of cable harness routing has also seen recent attention. Karlsson et al. (2024) formulated harness routing as a combination of the shortest-path problem and the Steiner tree problem, explicitly rewarding shared paths between cables to model the same physical reuse of an already installed segment, and reported solutions close to those of an exact solver at a fraction of its computational cost. Yang et al. (2024) combined A* with ant-colony optimisation for multi-branch wire-harness layout in aircraft, explicitly treating the problem as multi- objective search once several branch endpoints must be reached from a shared trunk, something analogous to RCRP’s multiple demand nodes.

Two points from the literature motivate the approach taken in this article. First, once a search must reach more than one endpoint, every study above modifies its underlying single-source, single-target algorithm rather than replacing it outright. This is the kind of modification RCRP requires. Second, several of these studies (Karlsson et al., 2024; Zhou et al., 2023) explicitly reward or exploit the reuse of an already installed segment instead of costing every traversal independently. This supports treating segment reuse as part of the cost model itself.

Also, the proposed state augmentation idea of tracking, alongside the current position, which of several required destinations have already been reached, has been investigated before, and it’s the principle underlying the classical Held-Karp dynamic program for subset based routing problems (Held & Karp, 1962) and the Dreyfus-Wagner dynamic program for the Steiner tree problem (Dreyfus & Wagner, 1971), on which RCRP can be seen as a restricted case, a fixed graph in which no new junction points may be introduced. A related construction builds Steiner trees directly from the overlap between shortest paths to different terminals (Fujita et al., 2024).

## 3 Proposed adaptation

## 3.1 Dijkstra’s algorithm

Dijkstra’s algorithm (Dijkstra, 1959) solves the single-source shortest-path problem: given an origin s, it finds the minimum cost route from s to a destination t in a non-negatively weighted graph, and it does so by finalising vertices one at a time in increasing order of distance from s. Every part of that mechanism is built around a single origin and a single destination: the algorithm’s stopping condition is “vertex t has been finalised”, and the state it reasons about at every step is simply “which vertex am I at, and at what cost”.

RCRP does not fit that description. The distribution panel s is a single origin, as before, but there is no single destination as the panel must be connected to every node in a demand set D, which for a conventional residence contains from seven to around forty electrical points. If Dijkstra’s algorithm is run unmodified with a set D of demand nodes, it will stop as soon as one demand node is reached and returns a route that says nothing about the rest of nodes in D; running it once per demand node and taking the union of the resulting routes does not solve RCRP either, because it lets each route be chosen independently of the others, so it cannot exploit the fact that two demand nodes may share the same stretch of conduit on the way from the panel.

The most important problem is before the stopping condition, and it is what the algorithm treats as a position. In the standard Dijkstra, two routes that reach the same vertex v are directly comparable on cost alone, and the algorithm keeps only the


cheaper one. This is the pruning rule that makes the algorithm efficient. Consider a vertex v reachable by two routes from the panel s: one route passes through a demand node d1 on the way to v at a total cost of 10, and a second route reaches v directly at a cost of 8, without passing through d1. Standard Dijkstra keeps only the second, cheaper route and discards the first. But for RCRP the first route is better because it has already connected d1, so continuing from v only needs to reach the remaining demand nodes, whereas the second route still owes a connection to d1 as well as to every demand node in the set D. Discarding the higher-cost route can therefore discard the only route that leads to the true minimum length solution. The same vertex, reached with different sets of demand nodes already connected, is not the same situation for RCRP even though standard Dijkstra treats it as one. Figure 1 illustrates this with the two routes P1 and P2 mentioned.

*Pl:s dl - v, total cost 10*

*P2:'s v (direct), total cost 8*

## Standard Dijkstra

Sees both routes simply as Keeps P2 (8) and discards P1 (10). losing the fact that 1 ws reached

## Proposed Approach

Sees (v, (d1) and (v, }): diferent

d1 and P2 P: has not.

already connected

*Fig. 1. Two routes from the panel s to the same vertex v. Route P1 reaches v at a cost of 10 after passing through the demand node d1, while route P2 reaches v directly at a cost of 8 without connecting d1. Standard Dijkstra keeps only P2 and discards P1, even though P1*

*is the route that has already connected a demand node.*

## 3.2 Multiple demand nodes adaptation

The first change was the goal itself. Standard Dijkstra’s implicit objective is v = t: the search is complete once a specific vertex has been explored. RCRP’s objective is not the identity of any single vertex but a coverage condition over a set: every demand node in D must have been connected to the panel by the returned tree. This is a strictly weaker requirement on any individual vertex, since no single vertex is privileged as the destination, and a strictly stronger requirement overall, since it must hold simultaneously for every member of D rather than for one chosen vertex.

This shift, from one objective to many, is what motivates the proposed adaptations. A search that only asks, “have I reached vertex t” has no way to represent “I have reached some nodes, but not all of D”. It therefore has no way to recognise when it is finished, nor to compare two partial routes fairly when they have made different amounts of progress toward covering D. Figure 2 contrasts the two goal conditions on the same underlying graph.


*Fig. 2. The two goal conditions on the same underlying graph. Standard Dijkstra stops as soon as the single destination t is finalised,*

*whereas the proposed adaptation stops only once every node of the demand set D has been connected to the panel s.*

## 3.3 State representation

In standard Dijkstra, a search state is simply a vertex v. That representation is sufficient for a single-objective search, but as mentioned in Section 3.1 it’s not sufficient here as it discards information (which demand nodes have already been connected) that RCRP needs to compare two routes correctly.

We redefine the state representation as the pair (v, S), where v is the current vertex, the frontier of the partial cable tree built so far, and S ⊆ D is the set of demand nodes already connected along the route that reached the state 𝜎. With this change, the state no longer represents the current position as it also handles the information of which nodes have been connected. Two routes that reach the same vertex v but with different sets S are different states, are not compared with each other, and are not allowed to prune one another.

With this representation, the goal condition becomes S = D, where a state is a solution once every demand node has been connected, regardless of which vertex the search happens to be sitting at when that happens. The pruning rule becomes: a new generated state (v, S) is discarded only if a state with the same key (v, S) was already explored at equal or lower guiding cost g (Section 3.4), meaning the same vertex and the same coverage level.

## 3.4 Cost criteria

The third change concerns how cost is measured, and it follows a simple physical fact that a purely graph-theoretic reading of RCRP can easily miss: a conduit segment, once wired to connect one demand node, is not consumed by that connection. The same physical cable run can serve as part of the route to a second, third, or later demand node without being installed again. A search that charges the full segment weight on every traversal will overestimate the true cable cost of the resulting tree, including in cases where it doubles back through an already wired junction to branch off toward a different demand node, as Figure 3 shows.

This approach separates two notions of cost that standard Dijkstra treats as one. The first is the search guiding cost 𝑔(𝜎), the plain sum of edge weights along the best route found so far to state 𝜎; this is the quantity that orders the priority queue. The second is the reported solution cost, computed only once a goal state has been reached and its tree reconstructed by following parent pointers back to the origin: rather than summing every edge traversal recorded along the path, the reported cost sums each distinct conduit segment in the reconstructed tree once. The search may cross a given segment twice, once to reach one branch and again while backing out to reach another, that segment is still counted only once, matching what it costs to install.

The two costs are not equivalent: Section 3.6 shows how they relate and what the search guarantees about the reported cost.


*Fig. 3. Reuse of an already wired conduit segment. The route reaches one demand node, doubles back through a junction that has already been wired and branches off towards a second demand node. The shared segment is installed once, so the reported cost counts it once*

*even though the search crosses it twice.*

## 3.5 Proposed algorithm

Algorithm 1 shows the full adaptation of the Dijkstra’s mechanism, and Figure 1 corresponds directly to its state key (v, S) and its pruning test. The algorithm keeps the state and the pruning rule of Section 3.3 and separates the search guiding cost g from the reported cable cost exactly as argued in Section 3.4, the main loop orders its priority queue on g alone, while the function ReconstructTree computes the reported cost from the set of distinct edges in the final tree

Algorithm 1: Multi-Demand Dijkstra

Input: Graph G = (V, E, w), origin s, demand set D

Output: Cable tree T connecting every node in D to s

Function MultiDemandDijkstra(G, s, D)

heap ← priority queue of (g, v, S) ordered by g

best ← empty map from (v, S) to the lowest g found so far

parent ← empty map from a state to its parent state

S0 ← {s} if s ∈ D, else ∅

best(s, S0) ← 0; Insert (0, s, S0) into heap

While heap is not empty do:

(g, v, S) ← ExtractMin(heap)

If g > best(v, S) then:

Continue

End

If S = D then:

Return ReconstructTree(parent, (v, S))

End

For each neighbour (v′, w′) of v in G do:

g′ ← g + w′


```
S′ ← S ∪ {v′} if v′ ∈ D, else S
If (v′, S′) ∉ best or g′ < best(v′, S′) then:
best(v′, S′) ← g′; parent(v′, S′) ← (v, S)
Insert (g′, v′, S′) into heap
End
End
End
Return "no solution" (D is unreachable from s)
Function ReconstructTree(parent, goalState)
edges ← empty sequence
While goalState has a parent do:
Prepend the edge crossed to edges
goalState ← parent(goalState)
End
uniqueEdges ← distinct edges in edges
cost ← sum of w(e) for e in uniqueEdges
Return (uniqueEdges, cost)
```

The vertex pruning rule of classical Dijkstra is now keyed on (v, S) rather than on v alone, which is what allows the two routes to vertex v in Figure 1 to coexist in the search instead of one silently eliminating the other. The value returned by ReconstructTree is the reported cost used throughout Section 4; the search guiding cost g is used only internally (to order the priority queue).

## 3.6 Optimality and validity of the pruning rule

The search handles two costs, introduced in Section 3.4. For a walk W that starts at s and may cross a conduit segment more than once, the guiding cost g(W) is the sum of the weights of all its edge crossings, and the reported cost c(W) is the sum of the weights of the distinct edges that W crosses. Hence c(W) ≤ g(W), with equality only if W crosses no segment twice. RCRP asks for a connected subgraph of minimum total weight that contains s and D, that is, a Steiner tree T* of cost c*. Let g* be the minimum of g over all walks from s that visit every node of D and let W* be the walk returned by the algorithm.

Proposition 1 (validity of the pruning rule for g). The algorithm returns a walk W* with g(W*) = g*. Justification: let (v, S) be the state of a partial walk. Because the weights are non-negative and the remaining task is only to reach the nodes of D \ S from v, the cheapest completion depends on (v, S) alone and not on the route followed so far. If two partial walks reach the same state with guiding costs g1 ≤ g2, any completion of the second can be attached to the first at no greater total cost, so discarding the second cannot discard every optimal walk. The search is therefore Dijkstra’s algorithm on the graph of states (v, S), where every transition has a non-negative weight, and the first goal state (S = D) extracted from the priority queue is optimal for g.

Proposition 2 (quality of the reported cost). c* ≤ c(W*) ≤ g* ≤ 2c*. Justification: the distinct edges of W* form a connected subgraph that contains s and every node of D, so c* ≤ c(W*), and c(W*) ≤ g(W*) = g* because c ≤ g for every walk. Conversely, a depth-first traversal of T* that starts at s crosses each edge of T* exactly twice and visits every node of D, so g* ≤ 2c*. If the distinct edges of W* contain a cycle, deleting one edge of the cycle keeps the subgraph connected and lowers its cost, so ReconstructTree can return a spanning tree of that subgraph without affecting the bound. The chain also gives c* ≥ g*/2, a lower bound available after every run, which certifies that the reported cost is at most 2c(W*)/g* times the optimum.

Minimising g does not in general minimise c, and the key (v, S) is not sufficient for pruning under c. When a walk is extended, the extra cost in c depends on which segments it has already installed, and two walks that reach the same state may have installed different segments. The following instance shows both effects. Let G have vertices s, h, d1 and d2, with D = {d1, d2} and edge weights w(s, h) = 6, w(h, d1) = 4, w(h, d2) = 4, w(s, d1) = 9 and w(d1, d2) = 7. The state (h, {d1}) is reached by s → d1 → h with g = 13 and distinct edges {s–d1, d1–h}, and by s → h → d1 → h with g = 14 and distinct edges {s–h, h– d1}. The rule of Section 3.3 keeps the first and discards the second. Extending each of them to d2 through h–d2 gives c = 9 + 4 + 4 = 17 for the first and c = 6 + 4 + 4 = 14 for the second, so the discarded state was the better one for c. The search in fact returns s → d1 → d2, with g* = 16 and c = 16, whereas the optimal tree {s–h, h–d1, h–d2} costs c* = 14. The excess of about 14% lies inside the factor of two of Proposition 2.


Two remedies restore exactness for c. The first adds the set of installed segments to the state, which removes the sufficiency problem but enlarges the state space from |V|·2^|D| to as many as |V|·2^|D|·2^|E| states in the worst case, so it is impractical beyond very small graphs. The second replaces the walk by a recursion over trees in which (v, S) stands for the cheapest tree that contains v and S, as in the Dreyfus-Wagner dynamic programme (Dreyfus & Wagner, 1971), which optimises c directly at a higher cost (Section 3.7). The adaptation proposed here keeps the walk-based search because its state space is smaller and its implementation simpler, and it accepts the guarantee of Proposition 2 in exchange. Throughout the article, “optimal” applied to the search refers to g, and the costs reported in Section 4 are costs of feasible layouts that are at most twice the optimum, not proofs of optimality.

## 3.7 Time and space complexity

The analysis below assumes that G is connected, so that |E| ≥ |V| − 1, and that the set S is stored as a bit mask of |D| bits, so that union, membership and equality tests take constant time for |D| up to the machine word length. There are at most |V|·2^|D| distinct states (v, S). A state is expanded only once, at its first extraction with the lowest guiding cost, and the expansion examines the edges incident to v, so the total number of heap insertions is at most 2|E|·2^|D|. With a binary heap, one insertion or extraction costs O(log(|E|·2^|D|)) = O(|D| + log |V|), since log |E| ≤ 2 log |V|. The running time is therefore O(2^|D| · |E| · (|D| + log |V|)). The maps visited and parent hold at most |V|·2^|D| entries each and the heap at most 2|E|·2^|D|, so the space is O(|E|·2^|D|); with a decrease-key heap it falls to O(|V|·2^|D|).

These are worst-case bounds, because the search stops at the first goal state. In the runs of Table 2, A2 explored 68,652,930 of the at most 38·2^22 ≈ 1.6×10^8 states (about 43%), and B2 explored 60,507,790 of the at most 61·2^29 ≈ 3.3×10^10 (about 0.2%). Peak memory in those two runs was between 136 and 154 bytes per explored state (9.33 GB and 9.34 GB), so 32 GB of RAM holds about 2×10^8 states. For A3 (|V| = 38, |D| = 38) and B4 (|V| = 61, |D| = 34) the bound is about 1.0×10^13 and 1.0×10^12 states, which is consistent with neither run completing.

The exponential dependence on |D| cannot be removed in general. RCRP is the Steiner tree problem in graphs with terminal set D ∪ {s}, which is NP-hard (Fujita et al., 2024). The Dreyfus-Wagner dynamic programme (Dreyfus & Wagner, 1971) solves that problem exactly, optimising the reported cost c directly, in O(3^|D|·|V| + 2^|D|·|V|² + |V|³) time. The walk-based search has the smaller base 2 in the exponent, in exchange for the weaker guarantee of Proposition 2 (Section 3.6).

## 4 Experimentation

The experimentation used two real residential floor plans (Martínez Álvarez, 2023). Each plan defines a fixed conduit graph, and on each one the proposed adaptation was run under several demand configurations, that is, different sets of electrical points that the distribution panel must reach. Every configuration was also solved by a human with expertise in cable routing, and that solution is used as the reference. Plan A is a single floor of a medium-sized residence, with 38 nodes and 50 conduit segments, and it was evaluated under three demand configurations, A1 to A3. Plan B is a larger floor plan, with 61 nodes and 90 conduit segments, and it was evaluated under four demand configurations, B1 to B4. Figures 4 and 5 show the two plans. In each figure, the upper panel is the architectural floor plan of the residence, and the lower panel is the conduit graph derived from it, with every node drawn at the position it occupies in the plan. Varying the demand set over a fixed graph separates the effect of the size of D from the effect of the size of the plan, which matters because the state space described in Section 3.3 grows with 2^|D| and not with the number of vertices alone. Table 1 lists the seven configurations. All graphs use non-negative edge weights representing physical conduit lengths.

*Table 1. Experiments defined over the two plans.*

| Plan |   | Configuration Total nodes |   |   |   | Edges |   | Demand nodes |D| |   | Human reference (m) |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A |   | A1 |   |   |   |   |   | 7 |   | 30.4 |   |
|   |   |   |   |   |   | 50 |   |   |   |   |   |
| A |   | A2 |   | 38 |   |   |   | 22 |   | 120.57 |   |
| A |   | A3 |   |   |   |   |   | 38 |   | 136.85 |   |
| B |   | B1 |   | 61 |   | 90 |   | 14 |   | 110.2 |   |
| B |   | B2 |   |   |   |   |   | 29 |   | 129.8 |   |


| Plan |   | Configuration Total nodes |   |   |   | Edges |   | Demand nodes |D| |   | Human reference (m) |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B |   | B3 |   |   |   |   |   | 14 |   | 130 |   |
| B |   | B4 |   |   |   |   |   | 34 |   | 171.5 |   |

*Fig. 4. Plan A (38 nodes, 50 conduit segments). Up: floor plan of the residence, showing the distribution panel and the electrical points.*

*Down: conduit graph of the same plan, with each node placed at its location in the floor plan.*


*Fig. 5. Plan B (61 nodes, 90 conduit segments). Up: floor plan of the residence, showing the distribution panel and the electrical points.*

*Down: conduit graph of the same plan, with each node placed at its location in the floor plan.*


## 4.1 Evaluation metrics

The following metrics are recorded for the proposed adaptation on each of the seven demand configurations:

- Solution cost (metres): total cable length of the returned tree, computed as the sum of distinct conduit segments as described in Section 3.4. Lower is better.

- Execution time (ms or s): duration from the algorithm start to solution return.

- States explored: number of (vertex, S) states popped from the priority queue.

- Memory usage (KB or MB): peak heap consumption during execution.

## 4.2 Hardware and implementation

All experiments were run on a desktop computer running Windows 11, with an AMD Ryzen 5 5600X processor and 32 GB of DDR4 RAM at 3200 MHz. The algorithm was implemented in Python and run as a single-threaded process. Timing was measured using the system timer, and peak memory using the tracemalloc module.

## 4.3 Reference methods

Besides the human reference solutions, two established methods are run on the same seven graphs and demand sets, so that the cost of the proposed adaptation can be set against methods with known guarantees. The first is the shortest-path heuristic of Takahashi and Matsuyama (1980) for the Steiner tree problem in graphs, which starts from the panel and repeatedly connects the nearest unconnected demand node to the current tree through a shortest path. It runs in polynomial time and its cost is at most twice the optimum. The second is an integer linear programme that selects the conduit segments to install under a directed-cut formulation of the Steiner tree problem (Wong, 1984; Koch & Martin, 1998), solved with a general-purpose solver [solver and version]. When it terminates within the time limit, it gives the optimal cost c* of Section 3.6 for a configuration, including A3 and B4, whose graphs have at most 61 nodes and 90 segments. For every method the metrics of Section 4.1 are recorded, and the gap to the optimum is computed as 100·(c − c*)/c*, for the proposed adaptation as well as for the human reference.

## 4.4 Results

Table 2 reports the performance of the proposed adaptation on the seven demand configurations; A3 and B4 are marked as not completed. Figures 6 to 10 compare the two layouts on each configuration that has been solved. In every figure, the left panel is the cable tree returned by the proposed adaptation and the right panel is the layout produced by the human expert, both drawn over the same floor plan.

*Table 2. Performance of the proposed adaptations on each experiment.*

| Experiment Demand |   |   |   | Cost (m) Human |   |   |   | Δ (%) |   | Time |   | States |   | Memory |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|   |   | nodes |D| |   |   |   | ref. (m) |   |   |   |   |   | explored |   |   |   |
| A1 |   | 7 |   | 26.60 |   | 30.4 |   | −12.5 |   | 4.33 ms |   | 310 |   | 88.7 KB |   |
| A2 |   | 22 |   | 105.65 |   |   | 120.57 −12.4 |   |   | 47 m 32 s |   | 68,652,930 |   | 9.33 GB |   |
| B1 |   | 14 |   | 99.20 |   | 110.2 |   | −10.0 |   | 9.41 s |   | 231,150 |   | 58.42 MB |   |
| B2 |   | 29 |   | 107.10 |   | 129.8 |   | −17.5 |   | 46 m 14 s |   | 60,507,790 |   | 9.34 GB |   |
| B3 |   | 14 |   | 125.70 |   | 130 |   | −3.3 |   | 9.79 s |   | 374,410 |   | 90.01 MB |   |
| A3 |   | 38 |   | Not completed |   | 136.85 – |   |   |   | – |   | – |   | – |   |
| B4 |   | 34 |   | Not completed |   | 171.5 |   | – |   | – |   | – |   | – |   |


The adaptation returned a complete cable tree in each of the five experiments it solved, and in all five the tree is shorter than the human reference: 26.60 m against 30.4 m on A1, 105.65 m against 120.57 m on A2, 99.20 m against 110.2 m on B1, 107.10 m against 129.8 m on B2 and 125.70 m against 130 m on B3. The Δ column expresses that difference as a percentage of the reference, so a negative value means less cable than the reference; the reductions range from 3.3% on B3 to 17.5% on B2.

Cost is not the quantity that varies most across the seven experiments. Execution time and memory are governed by the size of the demand set rather than by the size of the plan. A1, with 7 demand nodes, finishes in 4.33 ms after exploring 310 states, whereas A2, with 22 demand nodes on the same 38-node graph, takes 47 m 32 s and 9.33 GB after exploring more than 68 million states. B1 and B3 both have 14 demand nodes on the same 61-node graph and behave alike, close to 9.5 s and under 100 MB each, while B2, with 29 demand nodes on that graph, needs 46 m 14 s and 9.34 GB. This is the 2^|D| growth of the state space described in Section 3.3.

Experiments A3 and B4, with 38 and 34 demand nodes, did not complete on the machine described in Section 4.2 and are marked as “Not completed” in Table 2. The search was stopped after 12 h of execution without returning a solution, which is more than 15 times the longest completed run (47 m 32 s on A2). This is a practical cut-off, not a proof that the search cannot finish. At that point it had explored [number of states] states and used [peak memory] of memory. The worst-case bound of Section 3.7, about 1.0×10^13 states for A3 and 1.0×10^12 for B4, is consistent with this outcome, since the 32 GB of available memory holds about 2×10^8 states at the per-state cost observed on A2.

As with any comparison against a single set of reference solutions, a shorter result under this length-only objective is evidence of efficiency under that objective specifically, not a general claim that the algorithm’s routing plan is superior to the human solution in every respect a licensed electrician would need to satisfy. That qualification is discussed further in Section 5.

*Fig. 6. Experiment A1. Left (red): cable tree returned by the proposed adaptation, 26.60 m. Right (blue): human reference layout, 30.4 m.*

*Demand nodes are shown in red. The proposed tree is 3.8 m (12.5%) shorter.*


*Fig. 7. Experiment A2. Left (red): cable tree returned by the proposed adaptation, 105.65 m. Right (blue): human reference layout,*

*120.57 m. Demand nodes are shown in red. The proposed tree is 14.92 m (12.4%) shorter.*

*Fig. 8. Experiment B1. Left (red): cable tree returned by the proposed adaptation, 99.20 m. Right (blue): human reference layout, 110.2*

*m. Demand nodes are shown in red. The proposed tree is 11.0 m (10.0%) shorter.*

*Fig. 9. Experiment B2. Left (red): cable tree returned by the proposed adaptation, 107.10 m. Right (blue): human reference layout, 129.8*

*m. Demand nodes are shown in red. The proposed tree is 22.7 m (17.5%) shorter, the largest reduction of the five experiments.*

*Fig. 10. Experiment B3. Left (red): cable tree returned by the proposed adaptation, 125.70 m. Right (blue): human reference layout, 130 m. Demand nodes are shown in red. The proposed tree is only 4.3 m (3.3%) shorter, the smallest reduction of the five experiments.*

## 5 Discussion

The results in Section 4.4 support the central argument of Section 3: once Dijkstra’s algorithm is given a state that tracks connected demand nodes rather than only the current vertex, and once already-installed conduit segments are treated as


reusable rather than re-counted, the algorithm produces complete, connected solutions to RCRP and does so at lower cable length than the reused human reference in all five of the experiments the search completed, with reductions between 3.3% and 17.5%.

The size of the reduction does not follow the size of the plan or the size of the demand set. It is largest on B2, with 17.5%, and smallest on B3, with 3.3%, and B1 and B3 have the same number of demand nodes on the same graph yet differ by almost seven percentage points. What the reduction reflects is how much reusable conduit the human layout left unused on that instance, which is a property of the individual layout rather than of the size of the problem.

The cost of obtaining these solutions is the practical limit of the approach. Execution time and peak memory follow the 2^|D| growth of the state space, consistent with the growth rates reported for other multi-destination shortest-path formulations (Gao et al., 2025), from 4.33 ms and 88.7 KB at 7 demand nodes to roughly 47 minutes and 9.3 GB at 22 and at 29 demand nodes, and the two experiments with the largest demand sets returned no solution at all. A search of this form, exact for the guiding cost g, is therefore usable at the demand-set sizes of an individual circuit, not for a whole dwelling wired in a single pass.

However, there are some limitations that need to be discussed:

- The RCRP formulation treats every conduit segment as having unlimited cable-carrying capacity and does not model conductor gauge, circuit segregation, or the fill limits set by electrical codes such as the U.S. National Electrical Code (National Fire Protection Association, 2023). A solution that is optimal under the length-only objective evaluated here may not be installable as specified without further constraint checking; construction heuristics for the capacitated Steiner tree problem show one way such capacity limits could be incorporated directly into the search (Van den Eynde et al., 2022).

- The evaluation covers two single-floor residential plans of fewer than 65 nodes and seven demand configurations defined over them, of which only the five with 29 or fewer demand nodes were solved. The state space of Section 3.3 grows as 𝑂(|𝑉| × 2^|𝐷|) in the number of demand nodes, and residential floor plans keep |𝐷| in a low range; an entire building with many independently wired units, shared risers, and considerably larger demand sets is a different scale of problem.

- The search is exact only for the guiding cost g. The reported cost is guaranteed to be at most twice the optimum (Section 3.6), and the example given there shows that the excess can be non-zero, because the state (v, S) does not record which segments are already installed. The costs in Table 2 are therefore costs of feasible layouts, not proven optima, and the exact reference of Section 4.3 is the means to measure the remaining gap.

## 6 Conclusions

This article asked whether a classical shortest-path algorithm, Dijkstra’s algorithm, can be applied directly to the Residential Cable Routing Problem, and as it is, the answer is no, because as described before, Dijkstra’s algorithm is built for a single origin and a single destination, and RCRP requires connecting one origin to many demand nodes at once, which breaks both the algorithm’s stopping condition and the correctness of its usual pruning rule.

The proposed adaptation resolves this by redefining the search state as a vertex together with the set of demand nodes already connected, and by treating an already-installed conduit segment as reusable when computing the reported solution cost, while keeping the search’s own guiding cost a simple sum of edge weights, for which the pruning on the state key is valid. This yields a reported cost within a factor of two of the optimal tree, but not necessarily the optimal tree itself. Across seven demand configurations defined over two residential floor plans, the adapted algorithm produced a complete cable tree in the five it solved, and in all five the tree was shorter than the corresponding human reference, by between 3.3% and 17.5%. The two configurations with the largest demand sets returned no solution, which places the practical limit of the search near 30 demand nodes on graphs of this size.

Future work should pursue:

- Evaluation on residential floor plans beyond the two reused here.

- Reducing the gap between the guiding cost and the reported cost, for example by recording the installed segments in the state or by improving the returned tree with a Steiner post-optimisation step.

- Incorporating conduit fill capacity and electrical code constraints directly into the graph model, so that the objective reflects installable, not merely graph-optimal solutions.

- Extending the evaluation from single-family residential plans to larger, multi-unit buildings. Wiring an entire building is not the same problem, at the same scale, as wiring a single house: demand sets are larger, risers and shared trunks introduce additional structure that a flat conduit graph does not capture, and the 𝑂(|𝑉| × 2^|𝐷|) state space may no longer be manageable without further pruning or decomposition by floor or unit. Establishing whether this adaptation scales to that setting is left as future work.


## References

- Choi, W., Kim, C., Heo, S., & Na, S. (2022). The Modification of A* Pathfinding Algorithm for Building Mechanical, Electronic and Plumbing (MEP) Path. IEEE Access, 10, 65784–65800. https://doi.org/10.1109/ACCESS.2022.3184106

- Deng, R., Li, X., & Tian, Y. (2025). A Review: The Application of Path Optimization Algorithms in Building Mechanical, Electrical, and Plumbing Pipe Design. Buildings, 15(12), 2093. https://doi.org/10.3390/buildings15122093

- Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. Numerische Mathematik, 1(1), 269–271. https://doi.org/10.1007/BF01386390

- Dreyfus, S. E., & Wagner, R. A. (1971). The steiner problem in graphs. Networks, 1(3), 195–207. https://doi.org/10.1002/net.3230010302

- Fujita, M., Shimada, Y., Kimura, T., & Ikeguchi, T. (2024). A heuristic method for solving the Steiner tree problem in graphs using network centralities. PLOS ONE, 19(6), e0303764. https://doi.org/10.1371/journal.pone.0303764

- Gao, Y., Li, Z., He, J., Fang, J., Gao, H., Gao, X., & Chen, G. (2025). Algorithms for Shortest Path Tour Problem. Theoretical Computer Science, 1054, 115428. https://doi.org/10.1016/j.tcs.2025.115428

- Held, M., & Karp, R. M. (1962). A Dynamic Programming Approach to Sequencing Problems. Journal of the Society for Industrial and Applied Mathematics, 10(1), 196–210. https://doi.org/10.1137/0110015

- Karlsson, T., Åblad, E., Hermansson, T., Carlson, J. S., & Tenfält, G. (2024). Automatic Cable Harness Layout Routing in a Customizable 3D Environment. Computer-Aided Design, 169, 103671. https://doi.org/10.1016/j.cad.2023.103671

- Martínez Álvarez, M. (2023). Minimización de cableado de servicios residenciales aplicando algoritmos genéticos. https://ri.uaemex.mx/handle/20.500.11799/139966

- National Fire Protection Association. (2023). NFPA 70: National Electrical Code. NFPA.

- Van den Eynde, S., Audenaert, P., Colle, D., & Pickavet, M. (2022). A construction heuristic for the capacitated Steiner tree problem. PLOS ONE, 17(6), e0270147. https://doi.org/10.1371/journal.pone.0270147

- Yang, F., Wang, P., Zhang, R., Xing, S., Wang, Z., Li, M., & Fang, Q. (2024). A*–Ant Colony Optimization Algorithm for Multi-Branch Wire Harness Layout Planning. Electronics, 13(3), 529. https://doi.org/10.3390/electronics13030529

- Zhou, X., Sun, K., Wang, Q., Wang, J., Huang, X., & Zhou, W. (2023). IEDW: A BIM-based indoor electric distribution wiring algorithm using graph theory and capacity-limited multiple traveling salesman problem solver. Advanced Engineering Informatics, 56, 101999. https://doi.org/10.1016/j.aei.2023.101999
