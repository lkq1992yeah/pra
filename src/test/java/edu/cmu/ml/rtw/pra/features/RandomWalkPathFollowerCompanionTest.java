package edu.cmu.ml.rtw.pra.features;

import java.rmi.RemoteException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import junit.framework.TestCase;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;

import edu.cmu.graphchi.walks.distributions.DiscreteDistribution;
import edu.cmu.ml.rtw.pra.experiments.Instance;
import edu.cmu.ml.rtw.pra.graphs.GraphOnDisk;
import edu.cmu.ml.rtw.users.matt.util.FakeFileUtil;
import edu.cmu.ml.rtw.users.matt.util.TestUtil;
import edu.cmu.ml.rtw.users.matt.util.TestUtil.Function;

public class RandomWalkPathFollowerCompanionTest extends TestCase {

  FakeFileUtil fileUtil;
  GraphOnDisk graph;

  @Override
  public void setUp() {
    fileUtil = new FakeFileUtil();
    fileUtil.addFileToBeRead("/graph/node_dict.tsv", "1\tnode1\n");
    fileUtil.addFileToBeRead("/graph/edge_dict.tsv", "1\trel1\n");
    graph = new GraphOnDisk("/graph/", fileUtil);
  }

  public void testAcceptableRow() throws RemoteException {
    Set<Integer> allowedTargets = Sets.newHashSet();
    allowedTargets.add(2);
    allowedTargets.add(3);
    allowedTargets.add(10);
    final RandomWalkPathFollowerCompanion companion = new RandomWalkPathFollowerCompanion(
        graph,
        1,
        1024,
        new FakeVertexIdTranslate(),
        new PathType[1],
        MatrixRowPolicy.ALL_TARGETS,
        allowedTargets,
        true);
    final Set<Integer> sourceTargets = Sets.newHashSet();
    sourceTargets.add(2);
    final Set<Integer> allTargets = Sets.newHashSet(sourceTargets);
    allTargets.add(3);

    companion.setAcceptPolicy(MatrixRowPolicy.EVERYTHING);
    assertEquals(true, companion.acceptableRow(1, 20, sourceTargets, allTargets));

    companion.setAcceptPolicy(MatrixRowPolicy.ALL_TARGETS);
    assertEquals(true, companion.acceptableRow(1, 10, sourceTargets, allTargets));
    assertEquals(false, companion.acceptableRow(1, 20, sourceTargets, allTargets));
    companion.setAllowedTargets(null);
    assertEquals(false, companion.acceptableRow(1, 10, sourceTargets, allTargets));

    companion.setAcceptPolicy(MatrixRowPolicy.PAIRED_TARGETS_ONLY);
    assertEquals(true, companion.acceptableRow(1, 2, sourceTargets, allTargets));
    assertEquals(false, companion.acceptableRow(1, 3, sourceTargets, allTargets));

    companion.setAcceptPolicy(null);
    TestUtil.expectError(RuntimeException.class, new Function() {
      @Override
      public void call() {
        companion.acceptableRow(1, 2, sourceTargets, allTargets);
      }
    });
  }

  public void testGetFeatureMatrix() throws RemoteException {
    final RandomWalkPathFollowerCompanion companion = new RandomWalkPathFollowerCompanion(
        graph,
        1,
        1024,
        new FakeVertexIdTranslate(),
        new PathType[1],
        MatrixRowPolicy.EVERYTHING,  // so we don't have to worry about matching
        null,
        true);
    // We'll have one source, one path type, and two targets in this test.
    ConcurrentHashMap<Integer, ConcurrentHashMap<Integer, DiscreteDistribution>> dists =
        new ConcurrentHashMap<Integer, ConcurrentHashMap<Integer, DiscreteDistribution>>();
    int sourceNode = 1;
    int pathType = 1;
    int targetNode1 = 2;
    int targetNode2 = 3;
    DiscreteDistribution targetNodeDist = new DiscreteDistribution(
        new int[]{targetNode1, targetNode1, targetNode1, targetNode2});
    dists.put(sourceNode, new ConcurrentHashMap<Integer, DiscreteDistribution>());
    dists.get(sourceNode).put(pathType, targetNodeDist);
    companion.setDistributions(dists);

    List<Instance> instances = Lists.newArrayList();
    instances.add(new Instance(sourceNode, targetNode1, true, graph));
    instances.add(new Instance(sourceNode, targetNode2, true, graph));
    FeatureMatrix matrix = companion.getFeatureMatrix(instances);
    MatrixRow firstRow = new MatrixRow(instances.get(0),
                                       new int[]{pathType},
                                       new double[]{.75});
    MatrixRow secondRow = new MatrixRow(instances.get(1),
                                        new int[]{pathType},
                                        new double[]{.25});
    assertEquals(2, matrix.size());
    MatrixRow row = matrix.getRow(0);
    if (row.instance.target() == targetNode1) {
      assertEquals(1, row.columns);
      assertEquals(.75, row.values[0]);
    } else {
      assertEquals(targetNode2, row.instance.target());
      assertEquals(1, row.columns);
      assertEquals(.25, row.values[0]);
    }
    row = matrix.getRow(1);
    if (row.instance.target() == targetNode1) {
      assertEquals(1, row.columns);
      assertEquals(.75, row.values[0]);
    } else {
      assertEquals(targetNode2, row.instance.target());
      assertEquals(1, row.columns);
      assertEquals(.25, row.values[0]);
    }
  }
}
