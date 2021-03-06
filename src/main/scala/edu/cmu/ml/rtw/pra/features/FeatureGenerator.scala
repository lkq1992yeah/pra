package edu.cmu.ml.rtw.pra.features

import edu.cmu.ml.rtw.pra.experiments.Dataset

import scala.collection.JavaConverters._

trait FeatureGenerator {
  /**
   * Takes the data, probably does some random walks (or maybe some matrix multiplications, or a
   * few other possibilities), and returns a FeatureMatrix.
   */
  def createTrainingMatrix(data: Dataset): FeatureMatrix

  /**
   * For efficiency in creating the test matrix, we might drop some features if they have zero
   * weight.  In some FeatureGenerator implementations, computing feature values can be very
   * expensive, so this allows us to save some work.  The return value is the updated set of
   * weights, with any desired values removed.  Yes, this potentially changes the indices and thus
   * the meaning of the feature matrix.  Thus the updated weights can't be used anymore on the
   * training matrix, only on the test matrix.
   */
  def removeZeroWeightFeatures(weights: Seq[Double]): Seq[Double]

  /**
   * Constructs a matrix for the test data.  In general, if this step is dependent on training
   * (because, for instance, a feature set was selected at training time), the FeatureGenerator
   * should save that state internally, and use it to do this computation.  Not all implementations
   * need internal state to do this, but some do.
   */
  def createTestMatrix(data: Dataset): FeatureMatrix

  /**
   * Returns a string representation of the features in the feature matrix.  This need only be
   * defined after createTrainingMatrix is called once, and calling removeZeroWeightFeatures may
   * change the output of this function (because the training and test matrices may have different
   * feature spaces; see comments above).
   */
  def getFeatureNames(): Array[String]

  def createEdgesToExclude(data: Dataset, unallowedEdges: Seq[Int]): Seq[((Int, Int), Int)] = {
    // If there was no input data (e.g., if we are actually trying to predict new edges, not
    // just hide edges from ourselves to try to recover), then there aren't any edges to
    // exclude.  So return an empty list.
    if (data == null || unallowedEdges == null) {
      return Seq()
    }
    data.instances.flatMap(instance => {
      unallowedEdges.map(edge => {
        ((instance.source, instance.target), edge.toInt)
      })
    })
  }
}
