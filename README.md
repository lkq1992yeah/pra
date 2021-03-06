# PRA (Path Ranking Algorithm)

An implementation of the Path Ranking Algorithm (PRA) using GraphChi, a library for efficient
processing of large graphs on a single machine.  This algorithm learns models that analyze a graph
and predict missing edges of particular types.  The code here was used to run experiments in the
following papers:

* Incorporating Vector Space Similarity in Random Walk Inference over Knowledge Bases.  Matt
  Gardner, Partha Talukdar, Jayant Krishnamurthy, and Tom Mitchell.  EMNLP 2014.
([website](http://rtw.ml.cmu.edu/emnlp2014_vector_space_pra))
* Improving Learning and Inference in a Large Knowledge-base using Latent Syntactic Cues.  Matt
  Gardner, Partha Talukdar, Bryan Kisiel, and Tom Mitchell.  EMNLP 2013.
([website](http://rtw.ml.cmu.edu/emnlp2013_pra))

See [the github.io page](http://matt-gardner.github.io/pra/) for code documentation.  Please feel
free to file bugs, feature requests, or send pull requests.

# NOTE

This code generally takes quite a bit of memory.  That's probably a byproduct of how it was
developed; I typically use a machine that has 400GB of RAM, so I don't need to worry too much about
how much memory the code is using.  That means I probably do some things that are memory
inefficient; on NELL graphs, the code can easily use upwards of 40GB.  On larger graphs, and with
various parameter settings, it can easily use much more than that.  With small graphs, though, I
can successfully run the code on a machine that only has 8GB of RAM.  This needs some work to be
made more memory efficient on larger graphs.  It should be straightforward to implement a
stochastic gradient training regime, for instance, that would allow for much more memory-efficient
computation.

# License

This code makes use of a number of other libraries that are distributed under various open source
licenses (notably the Apache License and the Common Public License).  You can see those
dependencies listed in the build.sbt file.  The code under the src/ directory is distributed under
the terms of the GNU General Public License, version 3 (or, at your choosing, any later version of
that license).  You can find the text of that license
[here](http://www.gnu.org/licenses/gpl-3.0.txt).

# Changelog

Version 3.0 (released on 5/30/2015):

- More refinement on the parameter specification (hence the larger version bump, as the parameter
  files are not compatible with previous versions).  This nests parameters in the specification
file according to how they are used in the code, and makes some things in the code _way_ simpler.
I think the specification is also conceptually cleaner, but maybe someone else would just think
it's more verbose...

- A lot of code moved to scala, and in the process some of it became more configurable.

- It could still use some more versatility, but there are some improvements to how the graph
  works - there's a setting where you can keep the graph in memory, for instance, instead of using
GraphChi to do random walks over the graph on disk.  You can also make instance-specific graphs,
so that each training and testing instance has its own graph to use.  These need to be pretty
small for this to make sense, though.

- There is a new mechanism for selecting negative examples, using personalized page rank to select
  them instead of PRA's random walks.  It turns out that it doesn't affect PRA's performance at
all, really, but it allows for a better test scenario, and it allows for comparing methods on the
same training data, where some other method isn't capable of selecting its own negative examples.

- Allowed for other learning algorithms to use PRA's feature matrix.  We tried using SVMs with
  various kernels, and it turns out that logistic regression is better, at least on the metrics we
used.  And the code is set up to allow you to (relatively) easily experiment with other
algorithms, if you want to.

- Implemented a new way of creating a feature matrix over node pairs in the graph, which is
  simpler and easier than PRA; it's similar to just doing the first step of PRA and extracting a
feature matrix from the resulting subgraphs.  It's faster and works quite a bit better.

Version 2.0 (released on 3/4/2015):

- Much better parameter specification.  See [the github.io
  page](http://matt-gardner.github.io/pra) for information on the new way to specify and run
experiments.  This totally breaks backwards compatibility with older formats, so you'll need to go
read the documentation if you want to upgrade to this version.

- Working synthetic data generation.  There are a lot of parameters to play with here; see the
  documentation linked above for some more info.

- A matrix multiplication implementation of the vector space random walks from the EMNLP 2014
  paper.  This is at least done in theory.  I haven't gotten the performance to be quite as good
yet, but the mechanism for doing it is in the code.

- Better handling of JVM exit (version 1.1 and earlier tend to spit out InterruptedExceptions at
  you when it terminates, and most of the time won't give you back the sbt console).

Version 1.1 (released on 12/20/2014):

- ExperimentScorer now shows more information.  It used to only show each experiment ranked by an
  overall score (like MAP); now it does a significance test on those metrics, and shows a table of
each experiment's performance on each individual relation in the test.  ExperimentScorer is not
currently very configurable, though - you have to change the code if you want to show something
else.  This is relatively easy, though, as the parameters are all at the top of the file.  You
could also write another class that calls ExperimentScorer with your own parameters, if you want.

- Added matrix multiplication as an alternative to random walks in the path following step.  This
  is still somewhat experimental, and more details will be forthcoming in a few months.  There's a
new parameter that can be specified in the param file called `path follower`.  Set it to `matrix
multiplication` to use the new technique.  The value of this is mostly theoretical at this point,
as performance is pretty much identical to using random walks, except it's slower and less
scalable.  I plan on getting the vector space random walk idea into the matrix multiplication code
soon.

- Removed the onlyExplicitNegatives option, because it turns out it's redundant with a setting of
  the matrix accept policy.

- Started work on synthetic data generation, but it's not done yet (well, you can generate some
  data, but learning from it doesn't turn out as I expect.  Something is up...).  A final release
of working synthetic data generation will have to wait until version 1.2.
