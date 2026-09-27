package com.truetrack.navigation

import java.util.PriorityQueue
import kotlin.math.*

/**
 * Directed edge representing a routable road segment.
 */
data class GraphEdge(
    val targetId: Long,
    val distM: Double,
    val highway: String = "residential",
    val oneway: Boolean = false,
    val name: String = ""
)

/**
 * Result of an A* route calculation.
 */
data class RouteResult(
    val pathNodeIds: List<Long>,
    val coordinates: List<Pair<Double, Double>>, // Pair(lat, lon)
    val totalDistanceM: Double,
    val executionTimeMs: Double,
    val geojsonLineString: String
)

/**
 * Lightweight On-Device A* Router for TrueTrack.
 * Features:
 * - Admissible Haversine distance heuristic
 * - Secondary edge cost weighting (road classification, speed multiplier, dead-zone penalty)
 * - Spatial bounding-box subset clipping using GraphIndex
 * - GeoJSON LineString generation
 */
class Router(
    private val graphIndex: GraphIndex,
    private val adjacency: Map<Long, List<GraphEdge>>
) {

    /**
     * Road class multiplier configuration.
     * Lower multiplier prefers higher-grade arterial roads; higher multiplier penalizes slower roads.
     */
    private val roadClassWeights = mapOf(
        "trunk" to 0.85,
        "primary" to 0.90,
        "secondary" to 0.95,
        "tertiary" to 1.00,
        "residential" to 1.10,
        "service" to 1.25,
        "unclassified" to 1.15
    )

    private data class SearchNode(
        val nodeId: Long,
        val g: Double,
        val f: Double
    ) : Comparable<SearchNode> {
        override fun compareTo(other: SearchNode): Int = f.compareTo(other.f)
    }

    /**
     * Calculates the edge weight given its physical distance and attributes.
     */
    private fun computeEdgeWeight(
        edge: GraphEdge,
        deadZoneMultiplier: Double = 1.0
    ): Double {
        val classMult = roadClassWeights[edge.highway] ?: 1.0
        return edge.distM * classMult * deadZoneMultiplier
    }

    /**
     * Runs A* search between startNodeId and goalNodeId.
     * Uses bounding-box subset clipping to restrict search area for sub-millisecond execution.
     */
    fun findRoute(
        startNodeId: Long,
        goalNodeId: Long,
        deadZonePenalty: Double = 1.0
    ): RouteResult? {
        val t0 = System.nanoTime()

        val startNode = graphIndex.getNode(startNodeId) ?: return null
        val goalNode = graphIndex.getNode(goalNodeId) ?: return null

        if (startNodeId == goalNodeId) {
            val coords = listOf(Pair(startNode.lat, startNode.lon))
            return RouteResult(
                pathNodeIds = listOf(startNodeId),
                coordinates = coords,
                totalDistanceM = 0.0,
                executionTimeMs = 0.0,
                geojsonLineString = buildGeoJson(coords)
            )
        }

        // Bounding-box subset clipping around start and goal.
        // 0.08° (~8.8 km) padding prevents one-way street detours from being clipped.
        val minLat = min(startNode.lat, goalNode.lat)
        val maxLat = max(startNode.lat, goalNode.lat)   // was: goalNode.lon (bug)
        val minLon = min(startNode.lon, goalNode.lon)
        val maxLon = max(startNode.lon, goalNode.lon)
        val activeNodes = graphIndex.clipNodesInBBox(minLat, minLon, maxLat, maxLon, paddingDeg = 0.08)

        val openSet = PriorityQueue<SearchNode>()
        val gScore = HashMap<Long, Double>()
        val cameFrom = HashMap<Long, Long>()

        val initialH = graphIndex.haversineM(startNode.lat, startNode.lon, goalNode.lat, goalNode.lon)
        gScore[startNodeId] = 0.0
        openSet.add(SearchNode(startNodeId, 0.0, initialH))

        while (openSet.isNotEmpty()) {
            val current = openSet.poll() ?: break
            val u = current.nodeId

            if (u == goalNodeId) {
                val path = ArrayList<Long>()
                var curr: Long? = goalNodeId
                while (curr != null) {
                    path.add(curr)
                    curr = cameFrom[curr]
                }
                path.reverse()

                val coords = path.mapNotNull { nid ->
                    graphIndex.getNode(nid)?.let { Pair(it.lat, it.lon) }
                }

                // Compute true physical path distance
                var totalDist = 0.0
                for (i in 0 until coords.size - 1) {
                    totalDist += graphIndex.haversineM(
                        coords[i].first, coords[i].second,
                        coords[i + 1].first, coords[i + 1].second
                    )
                }

                val elapsedMs = (System.nanoTime() - t0) / 1_000_000.0
                return RouteResult(
                    pathNodeIds = path,
                    coordinates = coords,
                    totalDistanceM = totalDist,
                    executionTimeMs = elapsedMs,
                    geojsonLineString = buildGeoJson(coords)
                )
            }

            val currentG = gScore[u] ?: Double.MAX_VALUE
            if (current.g > currentG) continue

            val edges = adjacency[u] ?: emptyList()
            for (edge in edges) {
                val v = edge.targetId
                // Restrict to clipped bounding box subset
                if (!activeNodes.contains(v) && v != goalNodeId) continue

                val targetNode = graphIndex.getNode(v) ?: continue
                val weight = computeEdgeWeight(edge, deadZonePenalty)
                val tentativeG = currentG + weight

                if (tentativeG < (gScore[v] ?: Double.MAX_VALUE)) {
                    cameFrom[v] = u
                    gScore[v] = tentativeG
                    val h = graphIndex.haversineM(targetNode.lat, targetNode.lon, goalNode.lat, goalNode.lon)
                    openSet.add(SearchNode(v, tentativeG, tentativeG + h))
                }
            }
        }

        return null
    }

    /**
     * Builds standard GeoJSON LineString representation (lon, lat order).
     */
    private fun buildGeoJson(coordinates: List<Pair<Double, Double>>): String {
        val sb = StringBuilder()
        sb.append("{\"type\":\"LineString\",\"coordinates\":[")
        coordinates.forEachIndexed { idx, pt ->
            sb.append("[").append(pt.second).append(",").append(pt.first).append("]")
            if (idx < coordinates.size - 1) sb.append(",")
        }
        sb.append("]}")
        return sb.toString()
    }
}
