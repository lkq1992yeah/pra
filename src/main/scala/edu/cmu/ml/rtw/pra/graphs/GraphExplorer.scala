package edu.cmu.ml.rtw.pra.graphs

import edu.cmu.ml.rtw.pra.config.JsonHelper
import edu.cmu.ml.rtw.pra.config.PraConfig
import edu.cmu.ml.rtw.pra.experiments.Dataset
import edu.cmu.ml.rtw.pra.experiments.Instance
import edu.cmu.ml.rtw.pra.features.BasicPathTypeFactory
import edu.cmu.ml.rtw.pra.features.RandomWalkPathFinder
import edu.cmu.ml.rtw.pra.features.PathFinderCreator
import edu.cmu.ml.rtw.pra.features.PathType
import edu.cmu.ml.rtw.pra.features.PathTypePolicy
import edu.cmu.ml.rtw.pra.features.SingleEdgeExcluder
import edu.cmu.ml.rtw.users.matt.util.Pair

import scala.collection.JavaConverters._

import org.json4s._
import org.json4s.native.JsonMethods._

/**
 * This is similar to a FeatureGenerator, in that it does the same thing as the first step of PRA,
 * but it does not actually produce a feature matrix.  The idea here is just to see what
 * connections there are between a set of nodes in a graph, and that's it.
 */
class GraphExplorer(params: JValue, config: PraConfig, praBase: String) {
  val paramKeys = Seq("path finder")
  JsonHelper.ensureNoExtras(params, "pra parameters -> explore", paramKeys)

  def findConnectingPaths(data: Dataset): Map[Instance, Map[PathType, Int]] = {
    println("Finding connecting paths")

    val finder = PathFinderCreator.create(params \ "path finder", config, praBase)
    finder.findPaths(config, data, Seq())

    val pathCountMap = finder.getPathCountMap().asScala.mapValues(
      _.asScala.mapValues(_.toInt).toMap
    ).toMap
    finder.finished
    pathCountMap
  }
}
