package edu.cmu.ml.rtw.pra.features

import edu.cmu.ml.rtw.pra.config.PraConfigBuilder
import edu.cmu.ml.rtw.pra.experiments.Dataset
import edu.cmu.ml.rtw.pra.experiments.Instance
import edu.cmu.ml.rtw.pra.experiments.Outputter
import edu.cmu.ml.rtw.pra.graphs.GraphOnDisk
import edu.cmu.ml.rtw.users.matt.util.Dictionary
import edu.cmu.ml.rtw.users.matt.util.FakeFileUtil
import edu.cmu.ml.rtw.users.matt.util.Pair
import edu.cmu.ml.rtw.users.matt.util.TestUtil
import edu.cmu.ml.rtw.users.matt.util.TestUtil.Function

import org.scalatest._

import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods._

import scala.collection.JavaConverters._

class SubgraphFeatureGeneratorSpec extends FlatSpecLike with Matchers {
  type Subgraph = java.util.Map[PathType, java.util.Set[Pair[Integer, Integer]]]

  val params: JValue = ("include bias" -> true)
  val graph = new GraphOnDisk("src/test/resources/")
  val config = new PraConfigBuilder().setNoChecks()
    .setGraph(graph).build()
  val fakeFileUtil = new FakeFileUtil

  val generator = new SubgraphFeatureGenerator(params, "/", config, fakeFileUtil)
  generator.featureDict.getIndex("feature1")
  generator.featureDict.getIndex("feature2")
  generator.featureDict.getIndex("feature3")

  def getSubgraph(instance: Instance) = {
    val subgraph = new java.util.HashMap[PathType, java.util.Set[Pair[Integer, Integer]]]
    val pathType1 = new BasicPathTypeFactory().fromString("-1-")
    val pathType2 = new BasicPathTypeFactory().fromString("-2-")
    val nodePairs1 = new java.util.HashSet[Pair[Integer, Integer]]
    nodePairs1.add(Pair.makePair(Integer.valueOf(instance.source), 1:Integer))
    val nodePairs2 = new java.util.HashSet[Pair[Integer, Integer]]
    nodePairs2.add(Pair.makePair(Integer.valueOf(instance.target), 2:Integer))
    subgraph.put(pathType1, nodePairs1)
    subgraph.put(pathType2, nodePairs2)
    Map(instance -> subgraph)
  }

  val instance = new Instance(1, 2, true, graph)
  val dataset = new Dataset(Seq(instance))

  "createTrainingMatrix" should "return extracted features from local subgraphs" in {
    val subgraph = getSubgraph(instance)
    val featureMatrix = new FeatureMatrix(List[MatrixRow]().asJava)
    val generator = new SubgraphFeatureGenerator(params, "/", config, fakeFileUtil) {
      override def getLocalSubgraphs(data: Dataset) = {
        if (data != dataset) throw new RuntimeException()
        subgraph
      }
      override def extractFeatures(subgraphs: Map[Instance, Subgraph]) = {
        if (subgraphs != subgraph) throw new RuntimeException()
        featureMatrix
      }
    }
    generator.createTrainingMatrix(dataset) should be(featureMatrix)
  }

  "createTestMatrix" should "create the same thing as createTrainingMatrix, and output the matrix" in {
    val subgraph = getSubgraph(instance)
    val matrixRow = new MatrixRow(instance, Array(0, 1, 2), Array(1.0, 1.0, 1.0))
    val featureMatrix = new FeatureMatrix(List(matrixRow).asJava)
    val nodeDict = new Dictionary()
    nodeDict.getIndex("node1")
    nodeDict.getIndex("node2")
    val out = new Outputter(null, fakeFileUtil)
    val config = new PraConfigBuilder().setOutputMatrices(true)
      .setOutputBase("/").setOutputter(out).setNoChecks().build()
    val generator = new SubgraphFeatureGenerator(params, "/", config, fakeFileUtil) {
      override def getLocalSubgraphs(data: Dataset) = {
        if (data != dataset) throw new RuntimeException()
        subgraph
      }
      override def extractFeatures(subgraphs: Map[Instance, Subgraph]) = {
        if (subgraphs != subgraph) throw new RuntimeException()
        featureMatrix
      }
    }
    generator.hashFeature("feature1")
    generator.hashFeature("feature2")

    fakeFileUtil.onlyAllowExpectedFiles
    fakeFileUtil.addExpectedFileWritten("/test_matrix.tsv",
      "node1,node2\tbias,1.0 -#- feature1,1.0 -#- feature2,1.0\n")
    generator.createTestMatrix(dataset) should be(featureMatrix)
    fakeFileUtil.expectFilesWritten
  }

  it should "not output the matrix when the output dir is null" in {
    val subgraph = getSubgraph(instance)
    val featureMatrix = new FeatureMatrix(List[MatrixRow]().asJava)
    val generator = new SubgraphFeatureGenerator(params, "/", config, fakeFileUtil) {
      override def getLocalSubgraphs(data: Dataset) = {
        if (data != dataset) throw new RuntimeException()
        subgraph
      }
      override def extractFeatures(subgraphs: Map[Instance, Subgraph]) = {
        if (subgraphs != subgraph) throw new RuntimeException()
        featureMatrix
      }
    }
    generator.createTestMatrix(dataset) should be(featureMatrix)
  }

  "removeZeroWeightFeatures" should "not remove anything" in {
    val weights = Seq(1.0, 2.0, 3.0)
    generator.removeZeroWeightFeatures(weights) should be(weights)
  }

  "getFeatureNames" should "just return the strings in the featureDict, plus a bias feature" in {
    generator.getFeatureNames() should be(Array("bias", "feature1", "feature2", "feature3"))
  }

  "getLocalSubgraphs" should "find correct subgraphs on a simple graph" in {
    // Because this is is a randomized process, we just test for things that should show up pretty
    // much all of the time.  If this test fails occasionally, it might not necessarily mean that
    // something is broken.

    // And we're only checking for one training instance, because that's all there is in the
    // dataset.
    val subgraph = generator.getLocalSubgraphs(dataset)(instance)
    val factory = new BasicPathTypeFactory
    var pathType = factory.fromString("-1-")
    subgraph.get(pathType) should contain(Pair.makePair(1:Integer, 2:Integer))
    pathType = factory.fromString("-3-_3-")
    subgraph.get(pathType) should contain(Pair.makePair(1:Integer, 7:Integer))
    pathType = factory.fromString("-1-2-")
    subgraph.get(pathType) should contain(Pair.makePair(1:Integer, 3:Integer))
    pathType = factory.fromString("-2-")
    subgraph.get(pathType) should contain(Pair.makePair(2:Integer, 3:Integer))
    pathType = factory.fromString("-3-")
    subgraph.get(pathType) should contain(Pair.makePair(1:Integer, 4:Integer))
    pathType = factory.fromString("-3-4-")
    subgraph.get(pathType) should contain(Pair.makePair(1:Integer, 5:Integer))
  }

  "extractFeatures" should "run the feature extractors and return a feature matrix" in {
    val generator = new SubgraphFeatureGenerator(params, "/", config, fakeFileUtil) {
      override def createExtractors(params: JValue) = {
        Seq(new FeatureExtractor() {
          override def extractFeatures(instance: Instance, subgraph: Subgraph) = {
            Seq("feature1", "feature2").asJava
          }
        })
      }
    }
    val subgraph = getSubgraph(instance)
    val featureMatrix = generator.extractFeatures(subgraph)
    featureMatrix.size should be(1)
    val matrixRow = featureMatrix.getRow(0)
    val expectedMatrixRow = new MatrixRow(instance, Array(0, 1, 2), Array(1.0, 1.0, 1.0))
    matrixRow.instance should be(expectedMatrixRow.instance)
    matrixRow.columns should be(expectedMatrixRow.columns)
    matrixRow.values should be(expectedMatrixRow.values)
  }

  "createExtractors" should "create PraFeatureExtractors correctly" in {
    val params: JValue = ("feature extractors" -> List("PraFeatureExtractor"))
    val generator = new SubgraphFeatureGenerator(params, "/", config, fakeFileUtil)
    generator.featureExtractors(0).getClass should be(classOf[PraFeatureExtractor])
  }

  it should "create OneSidedFeatureExtractors correctly" in {
    val params: JValue = ("feature extractors" -> List("OneSidedFeatureExtractor"))
    val generator = new SubgraphFeatureGenerator(params, "/", config, fakeFileUtil)
    generator.featureExtractors(0).getClass should be(classOf[OneSidedFeatureExtractor])
  }

  it should "fail on unrecognized feature extractors" in {
    val params: JValue = ("feature extractors" -> List("non-existant extractor"))
    TestUtil.expectError(classOf[IllegalStateException], "Unrecognized feature extractor", new Function() {
      def call() {
        val generator = new SubgraphFeatureGenerator(params, "/", config, fakeFileUtil)
      }
    })
  }

  "hashFeature" should "use an identity hash if hashing is not enabled" in {
    generator.hashFeature("feature1") should be(1)
    generator.hashFeature("feature2") should be(2)
    generator.hashFeature("feature3") should be(3)
  }

  it should "correctly hash to the feature size when hashing is enabled" in {
    val params: JValue = ("feature size" -> 10)
    val generator = new SubgraphFeatureGenerator(params, "/", config, fakeFileUtil)
    val hash7 = generator.featureDict.getIndex("hash-7")
    val hash2 = generator.featureDict.getIndex("hash-2")
    val string1 = "a"  // hash code is 97
    generator.hashFeature(string1) should be(hash7)
    val string2 = " "  // hash code is 32
    generator.hashFeature(string2) should be(hash2)
    val string3 = "asdfasdf"  // hash code is -802263448
    generator.hashFeature(string3) should be(hash2)
  }

  "createMatrixRow" should "set feature values to 1 and add a bias feature" in {
    val expected = new MatrixRow(instance, Array(0, 3, 2, 1), Array(1.0, 1.0, 1.0, 1.0))
    val matrixRow = generator.createMatrixRow(instance, Seq(3, 2, 1))
    matrixRow.instance should be(expected.instance)
    matrixRow.columns should be(expected.columns)
    matrixRow.values should be(expected.values)
  }
}
