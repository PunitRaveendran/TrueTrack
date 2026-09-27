package com.truetrack.navigation

import kotlin.math.*

/**
 * Data class representing a road network node.
 */
data class GraphNode(
    val id: Long,
    val lat: Double,
    val lon: Double
)

/**
 * Spatial Grid-Bucket Index for the road network graph.
 * Provides O(1) grid cell lookup and fast spatial radius queries on device
 * without scanning the entire graph.
 */
class GraphIndex(
    private val cellSizeDeg: Double = 0.005
) {
    private val buckets = HashMap<Pair<Int, Int>, MutableList<GraphNode>>()
    private val nodeMap = HashMap<Long, GraphNode>()

    private fun cellKey(lat: Double, lon: Double): Pair<Int, Int> {
        val r = floor(lat / cellSizeDeg).toInt()
        val c = floor(lon / cellSizeDeg).toInt()
        return Pair(r, c)
    }

    /**
     * Builds the spatial index from a collection of GraphNodes.
     */
    fun build(nodes: Collection<GraphNode>) {
        buckets.clear()
        nodeMap.clear()
        for (node in nodes) {
            nodeMap[node.id] = node
            val key = cellKey(node.lat, node.lon)
            buckets.getOrPut(key) { ArrayList() }.add(node)
        }
    }

    fun getNode(id: Long): GraphNode? = nodeMap[id]

    /**
     * Haversine distance in meters between two lat/lon coordinates.
     */
    fun haversineM(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val rEarth = 6378137.0
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a = sin(dLat / 2.0).pow(2.0) +
                cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) *
                sin(dLon / 2.0).pow(2.0)
        return 2.0 * rEarth * atan2(sqrt(a), sqrt(1.0 - a))
    }

    /**
     * Queries all nodes within radiusM of the target position.
     */
    fun queryRadius(targetLat: Double, targetLon: Double, radiusM: Double): List<Pair<GraphNode, Double>> {
        val latDelta = radiusM / 110600.0
        val lonDelta = radiusM / (111320.0 * max(0.1, cos(Math.toRadians(targetLat))))

        val minR = floor((targetLat - latDelta) / cellSizeDeg).toInt()
        val maxR = floor((targetLat + latDelta) / cellSizeDeg).toInt()
        val minC = floor((targetLon - lonDelta) / cellSizeDeg).toInt()
        val maxC = floor((targetLon + lonDelta) / cellSizeDeg).toInt()

        val results = ArrayList<Pair<GraphNode, Double>>()
        for (r in minR..maxR) {
            for (c in minC..maxC) {
                val cellNodes = buckets[Pair(r, c)] ?: continue
                for (node in cellNodes) {
                    val dist = haversineM(targetLat, targetLon, node.lat, node.lon)
                    if (dist <= radiusM) {
                        results.add(Pair(node, dist))
                    }
                }
            }
        }
        results.sortBy { it.second }
        return results
    }

    /**
     * Finds the nearest node to (targetLat, targetLon) within maxRadiusM.
     */
    fun findNearestNode(targetLat: Double, targetLon: Double, maxRadiusM: Double = 1000.0): GraphNode? {
        val candidates = queryRadius(targetLat, targetLon, maxRadiusM)
        return candidates.firstOrNull()?.first
    }

    /**
     * Returns the set of node IDs falling within a padded bounding box.
     */
    fun clipNodesInBBox(
        minLat: Double, minLon: Double,
        maxLat: Double, maxLon: Double,
        paddingDeg: Double = 0.003
    ): Set<Long> {
        val minR = floor((minLat - paddingDeg) / cellSizeDeg).toInt()
        val maxR = floor((maxLat + paddingDeg) / cellSizeDeg).toInt()
        val minC = floor((minLon - paddingDeg) / cellSizeDeg).toInt()
        val maxC = floor((maxLon + paddingDeg) / cellSizeDeg).toInt()

        val subset = HashSet<Long>()
        for (r in minR..maxR) {
            for (c in minC..maxC) {
                val cellNodes = buckets[Pair(r, c)] ?: continue
                for (node in cellNodes) {
                    if (node.lat in (minLat - paddingDeg)..(maxLat + paddingDeg) &&
                        node.lon in (minLon - paddingDeg)..(maxLon + paddingDeg)
                    ) {
                        subset.add(node.id)
                    }
                }
            }
        }
        return subset
    }
}
